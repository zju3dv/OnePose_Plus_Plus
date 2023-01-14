from itertools import chain
import h5py
from ray.actor import ActorHandle
from tqdm import tqdm
import json
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from loguru import logger
import ray
import math

from collections import defaultdict
from pathlib import Path

from src.utils.colmap import read_write_model
from src.utils.ray_utils import ProgressBar, chunks, chunk_index, split_dict

cfgs = {
    "ray": {
        "slurm": False,
        "n_workers": 4,
        "n_cpus_per_worker": 1,
        "n_gpus_per_worker": 0.25,
        "local_mode": False,
    },
}


def get_box_path(img_path):
    return img_path.replace("color", "bbox").replace(".png", ".txt")


def get_pose_path(img_path):
    return img_path.replace("color", "poses").replace(".png", ".txt")


def get_default_path(cfg, outputs_dir):
    deep_sfm_dir = osp.join(outputs_dir, "sfm_ws")
    model_dir = osp.join(deep_sfm_dir, "model")
    anno_dir = osp.join(outputs_dir, "anno")

    Path(anno_dir).mkdir(exist_ok=True, parents=True)

    return model_dir, anno_dir


def inverse_id_name(images):
    """ traverse keys of images.bin({id: image_name}), get {image_name: id} mapping. """
    inverse_dict = {}
    for key in images.keys():
        img_name = images[key].name

        inverse_dict[img_name] = key

    return inverse_dict


def id_mapping(points_idxs):
    """ traverse points_idxs({new_3dpoint_id: old_3dpoints_idxs}), get {old_3dpoint_idx: new_3dpoint_idx} mapping. """
    kp3d_id_mapping = {}  # {old_point_idx: new_point_idx}

    for new_point_idx, old_point_idxs in points_idxs.items():
        for old_point_idx in old_point_idxs:
            assert old_point_idx not in kp3d_id_mapping.keys()
            kp3d_id_mapping[old_point_idx] = new_point_idx

    return kp3d_id_mapping


def gather_3d_anno(
    keypoints_2d,
    descriptors_2d,
    scores_2d,
    kp3d_ids,
    feature_idxs,
    kp3d_id_feature,
    kp3d_id_score,
):
    """ For each 3d point, gather all corresponding 2d information """
    kp3d_id_to_kp2d_idx = {}
    for kp3d_id, feature_idx in zip(kp3d_ids, feature_idxs):
        kp3d_id_to_kp2d_idx[kp3d_id] = feature_idx

        if kp3d_id not in kp3d_id_feature:
            kp3d_id_feature[kp3d_id] = descriptors_2d[:, feature_idx][None]
            kp3d_id_score[kp3d_id] = scores_2d[feature_idx][None]
        else:
            kp3d_id_feature[kp3d_id] = np.append(
                kp3d_id_feature[kp3d_id], descriptors_2d[:, feature_idx][None], axis=0
            )
            kp3d_id_score[kp3d_id] = np.append(
                kp3d_id_score[kp3d_id], scores_2d[feature_idx][None], axis=0
            )

    return kp3d_id_feature, kp3d_id_score, kp3d_id_to_kp2d_idx


def read_features(feature):
    """ decouple keypoints, descriptors and scores from feature """
    keypoints_2d = feature["keypoints"].__array__()
    descriptors_2d = feature["descriptors"].__array__()
    scores_2d = feature["scores"].__array__()

    return keypoints_2d, descriptors_2d, scores_2d


def count_features(img_lists, features, images, kp3d_id_mapping, verbose=True):
    """ Search for valid 2d-3d correspondences; Count 3d features. """
    kp3d_idx_feature = {}  # {new_3d_point_idx: [feature1, feature2, ...]}
    kp3d_idx_score = {}  # {new_3d_point_idx: [score1, score2, ...]}
    kp3d_idx_to_img_kp2d_idx = defaultdict(
        dict
    )  # {new_3d_point_idx: {image_name: 2d_point_idx}}

    inverse_dict = inverse_id_name(images)  # {image_name: id}
    # traverse each image to find valid 2d-3d correspondence
    logger.info("Count features begin...")
    if verbose:
        iter_obj = tqdm(img_lists)
    else:
        iter_obj = img_lists

    for img_name in iter_obj:
        feature = features[img_name]
        keypoints_2d, descriptors_2d, scores_2d = read_features(feature)
        feature_dim = descriptors_2d.shape[0]

        id_ = inverse_dict[img_name]
        image_info = images[id_]
        point3D_ids = image_info.point3D_ids

        filter_feature_idxs = (
            []
        )  # record valid 2d point idxs. Each of these 2d points have a correspondence with a 3d point.
        filter_kp3d_ids = (
            []
        )  # record valid 3d point idxs. Each of these 3d points have a correspondence with a 2d point in this image.
        feature_idxs = (
            np.where(point3D_ids != -1)[0] if np.any(point3D_ids != -1) else None
        )
        if feature_idxs is None:
            kp3d_idx_to_img_kp2d_idx[img_name] = {}
        else:
            for feature_idx in feature_idxs:
                kp3d_id = point3D_ids[feature_idx]
                if (
                    kp3d_id in kp3d_id_mapping.keys()
                ):  # select 3d points which are kept after filter
                    filter_kp3d_ids.append(kp3d_id)
                    filter_feature_idxs.append(feature_idx)

            kp3d_idx_feature, kp3d_idx_score, kp3d_idx_to_kp2d_idx = gather_3d_anno(
                keypoints_2d,
                descriptors_2d,
                scores_2d,
                filter_kp3d_ids,
                filter_feature_idxs,
                kp3d_idx_feature,
                kp3d_idx_score,
            )

            kp3d_idx_to_img_kp2d_idx[img_name] = kp3d_idx_to_kp2d_idx

    return feature_dim, kp3d_idx_feature, kp3d_idx_score, kp3d_idx_to_img_kp2d_idx


def collect_descriptors(descriptors, feature_dim, num_leaf):
    num_descriptors = descriptors.shape[0]
    if num_descriptors < num_leaf:
        ret_descriptors = np.append(
            descriptors, np.ones((feature_dim, num_leaf - num_descriptors))
        )
    else:
        ret_descriptors = descriptors[:num_leaf]

    return ret_descriptors


def collect_scores(scores, num_leaf):
    num_scores = scores.shape[0]
    if num_scores < num_leaf:
        ret_scores = np.append(scores, np.zeros((num_leaf - num_scores, 1)), axis=0)
    else:
        ret_scores = scores[:num_leaf]

    return ret_scores


def collect_3d_ann_v2(
    kp3d_id_feature, kp3d_id_score, xyzs, points_idxs, feature_dim, num_leaf
):
    kp3d_position = np.empty(shape=(0, 3))
    kp3d_descriptors = np.empty(shape=(0, feature_dim))
    kp3d_scores = np.empty(shape=(0, 1))

    for new_point_idx, old_points_idxs in points_idxs.items():
        descriptors = np.empty(shape=(0, feature_dim))
        scores = np.empty(shape=(0, 1))
        for old_point_idx in old_points_idxs:
            descriptors = np.append(descriptors, kp3d_id_feature[old_point_idx], axis=0)
            scores = np.append(
                scores, kp3d_id_score[old_point_idx].reshape(-1, 1), axis=0
            )

        descriptors = collect_descriptors(descriptors, feature_dim, num_leaf)
        scores = collect_scores(scores, num_leaf)

        kp3d_position = np.append(
            kp3d_position, xyzs[new_point_idx].reshape(1, 3), axis=0
        )
        kp3d_descriptors = np.append(kp3d_descriptors, descriptors, axis=0)
        kp3d_scores = np.append(kp3d_scores, scores, axis=0)

    assert kp3d_descriptors.shape[0] == kp3d_position.shape[0] * num_leaf
    assert kp3d_scores.shape[0] == kp3d_position.shape[0] * num_leaf

    return kp3d_position, kp3d_descriptors, kp3d_scores


def average_3d_ann(kp3d_id_feature, kp3d_id_score, xyzs, points_idxs, feature_dim):
    """ 
    average position, descriptors and scores for 3d points 
    new_point_feature = avg(all merged 3d points features) = avg(all matched 2d points features)
    """
    kp3d_descriptors = np.empty(shape=(0, feature_dim))
    kp3d_scores = np.empty(shape=(0, 1))
    kp3d_position = np.empty(shape=(0, 3))

    # idx = 0
    # kp_position_mapping = {} # {point idx in kp3d_position: idx in filtered 3d points}
    for new_point_idx, old_points_idxs in points_idxs.items():
        descriptors = np.empty(shape=(0, feature_dim))
        scores = np.empty(shape=(0, 1))
        for old_point_idx in old_points_idxs:
            descriptors = np.append(descriptors, kp3d_id_feature[old_point_idx], axis=0)
            scores = np.append(
                scores, kp3d_id_score[old_point_idx].reshape(-1, 1), axis=0
            )

        avg_descriptor = np.mean(descriptors, axis=0).reshape(1, -1)
        avg_score = np.mean(scores, axis=0).reshape(1, -1)

        kp3d_position = np.append(
            kp3d_position, xyzs[new_point_idx].reshape(1, 3), axis=0
        )

        kp3d_descriptors = np.append(kp3d_descriptors, avg_descriptor, axis=0)
        kp3d_scores = np.append(kp3d_scores, avg_score, axis=0)

    return kp3d_position, kp3d_descriptors, kp3d_scores


def gather_3d_ann(
    kp3d_id_feature,
    kp3d_id_score,
    xyzs,
    points_idxs,
    pba: ActorHandle = None,
    verbose=True,
):
    """ 
    Gather affiliated 2d points' positions, (mean/concated)descriptors and scores for each 3d points
    """
    kp3d_descriptors = None
    kp3d_scores = np.empty(shape=(0, 1))
    kp3d_position = np.empty(shape=(0, 3))
    idxs = []

    if verbose:
        if pba is None:
            logger.info("Gather 3D ann begin...")
            points_idxs = tqdm(points_idxs.items())
        else:
            points_idxs = points_idxs.items()
    else:
        assert pba is None
        points_idxs = points_idxs.items()

    for new_point_idx, old_points_idxs in points_idxs:
        descriptors = None
        scores = np.empty(shape=(0, 1))
        for old_point_idx in old_points_idxs:
            descriptors = (
                np.append(descriptors, kp3d_id_feature[old_point_idx], axis=0)
                if descriptors is not None
                else kp3d_id_feature[old_point_idx]
            )
            scores = np.append(
                scores, kp3d_id_score[old_point_idx].reshape(-1, 1), axis=0
            )

        kp3d_position = np.append(
            kp3d_position, xyzs[new_point_idx].reshape(1, 3), axis=0
        )
        kp3d_descriptors = (
            np.append(kp3d_descriptors, descriptors, axis=0)
            if kp3d_descriptors is not None
            else descriptors
        )
        kp3d_scores = np.append(kp3d_scores, scores, axis=0)
        idxs.append(descriptors.shape[0] if descriptors is not None else 0)

        if pba is not None:
            pba.update.remote(1)

    return kp3d_position, kp3d_descriptors, kp3d_scores, np.array(idxs)


@ray.remote(num_cpus=1)
def gather_3d_ann_ray_wrapper(*args, **kwargs):
    return gather_3d_ann(*args, **kwargs)


def save_3d_anno(xyzs, descriptors, scores, out_path):
    """ Save 3d anno for each object """
    descriptors = descriptors.transpose(1, 0)
    np.savez(out_path, keypoints3d=xyzs, descriptors3d=descriptors, scores3d=scores)


def get_assign_matrix(xys, xyzs, kp3d_idx_to_kp2d_idx, kp3d_id_mapping, verbose=True):
    """ 
    Given 2d-3d correspondence(n pairs), build assign matrix(2*n) for this image 
    @param xys: all 2d keypoints extracted in this image.
    @param xyzs: all 3d points after filter.
    @param kp3d_idx_to_kp2d_idx: valid 2d-3d correspondences in this image. {kp3d_idx: kp2d_idx}
    @param kp3d_id_mapping: {3dpoint_before_filter_idx: 3dpoint_after_filter_idx}
    """
    kp2d_idxs = np.arange(len(xys))
    kp3d_idxs = np.arange(len(xyzs))

    MN1 = []
    for idx3d, idx2d in kp3d_idx_to_kp2d_idx.items():
        assert idx3d in kp3d_id_mapping.keys()
        new_idx3d = kp3d_id_mapping[idx3d]
        new_idx2d = idx2d

        if new_idx3d not in kp3d_idxs:
            kp2d_idxs = np.delete(kp2d_idxs, np.where(kp2d_idxs == new_idx2d))
            continue

        assert new_idx2d in kp2d_idxs and new_idx3d in kp3d_idxs
        kp2d_idxs = np.delete(kp2d_idxs, np.where(kp2d_idxs == new_idx2d))
        kp3d_idxs = np.delete(kp3d_idxs, np.where(kp3d_idxs == new_idx3d))

        MN1.append([new_idx2d, new_idx3d])

    num_matches = len(MN1)
    assign_matrix = np.array(MN1).T
    total_2d_kpts = xys.shape[0]
    return num_matches, assign_matrix, total_2d_kpts


def save_2d_anno_for_each_image(
    cfg,
    img_path,
    keypoints_2d,
    descriptors_2d,
    scores_2d,
    assign_matrix,
    num_matches,
    save_feature=True,
):
    data_dir = osp.dirname(osp.dirname(img_path))
    anno_dir = osp.join(data_dir, f"anno_{cfg.network.detection}")
    Path(anno_dir).mkdir(exist_ok=True, parents=True)

    img_name = osp.basename(img_path)
    anno_2d_path = osp.join(anno_dir, img_name.replace(".png", ".json"))

    anno_2d = {
        "keypoints2d": keypoints_2d.tolist(),  # [n, 2]
        "scores2d": scores_2d.reshape(-1, 1).tolist(),  # [n, 1]
        "assign_matrix": assign_matrix.tolist(),  # [2, k]
        "num_matches": num_matches,
    }

    if save_feature:
        anno_2d.update({"descriptors2d": descriptors_2d.tolist()})
    with open(anno_2d_path, "w") as f:
        json.dump(anno_2d, f)

    return anno_2d_path


def save_2d_anno_dict(
    cfg,
    img_lists,
    features,
    filter_xyzs,
    points_idxs,
    img_kp3d_idx_to_kp2d_idx,
    anno2d_out_path,
    verbose=True,
):
    annotations = []
    anno_id = 0

    kp3d_id_mapping = id_mapping(points_idxs)
    if verbose:
        iter_obj = tqdm(img_lists, total=len(img_lists))
    else:
        iter_obj = img_lists

    for img_path in iter_obj:
        feature = features[img_path]
        kp3d_idx_to_kp2d_idx = img_kp3d_idx_to_kp2d_idx[img_path]

        keypoints_2d, descriptors_2d, scores_2d = read_features(feature)
        num_matches, assign_matrix = get_assign_matrix(
            keypoints_2d, filter_xyzs, kp3d_idx_to_kp2d_idx, kp3d_id_mapping
        )

        if num_matches != 0:
            anno_2d_path = save_2d_anno_for_each_image(
                cfg,
                img_path,
                keypoints_2d,
                descriptors_2d,
                scores_2d,
                assign_matrix,
                num_matches,
            )
            pose_path = get_pose_path(img_path)
            anno_id += 1
            annotation = {
                "anno_id": anno_id,
                "anno_file": anno_2d_path,
                "img_file": img_path,
                "pose_file": pose_path,
            }
            annotations.append(annotation)

    with open(anno2d_out_path, "w") as f:
        json.dump(annotations, f)


def save_2d_anno(
    cfg,
    img_lists,
    features,
    filter_xyzs,
    points_idxs,
    img_kp3d_idx_to_kp2d_idx,
    anno2d_out_path,
    save_feature_for_each_img=True,
    save_threshold=0.05,
    verbose=True,
):
    """ Save 2d annotations for each image and gather all 2d annotations """
    annotations = []
    anno_id = 0

    kp3d_id_mapping = id_mapping(points_idxs)

    logger.info("Save 2D anno begin...")
    if verbose:
        iter_obj = tqdm(img_lists, total=len(img_lists))
    else:
        iter_obj = img_lists

    for img_path in iter_obj:
        feature = features[img_path]
        kp3d_idx_to_kp2d_idx = img_kp3d_idx_to_kp2d_idx[img_path]

        keypoints_2d, descriptors_2d, scores_2d = read_features(feature)
        num_matches, assign_matrix, total_2d_kpts = get_assign_matrix(
            keypoints_2d,
            filter_xyzs,
            kp3d_idx_to_kp2d_idx,
            kp3d_id_mapping,
            verbose=verbose,
        )

        if num_matches > save_threshold * total_2d_kpts:
            anno_2d_path = save_2d_anno_for_each_image(
                cfg,
                img_path,
                keypoints_2d,
                descriptors_2d,
                scores_2d,
                assign_matrix,
                num_matches,
                save_feature=save_feature_for_each_img,
            )
            pose_path = get_pose_path(img_path)
            anno_id += 1
            annotation = {
                "anno_id": anno_id,
                "anno_file": anno_2d_path,
                "img_file": img_path,
                "pose_file": pose_path,
            }
            annotations.append(annotation)

    with open(anno2d_out_path, "w") as f:
        json.dump(annotations, f)


def mean_descriptors(descriptors, idxs):
    cumsum_idxs = np.cumsum(idxs)
    pre_cumsum_idxs = np.cumsum(idxs)[:-1]
    pre_cumsum_idxs = np.insert(pre_cumsum_idxs, 0, 0)

    descriptors_instance = [
        np.mean(descriptors[start:end], axis=0).reshape(1, -1)
        for start, end in zip(pre_cumsum_idxs, cumsum_idxs)
    ]
    avg_descriptors = np.concatenate(descriptors_instance, axis=0)
    return avg_descriptors


def mean_scores(scores, idxs):
    cumsum_idxs = np.cumsum(idxs)
    pre_cumsum_idxs = np.cumsum(idxs)[:-1]
    pre_cumsum_idxs = np.insert(pre_cumsum_idxs, 0, 0)

    scores_instance = [
        np.mean(scores[start:end], axis=0).reshape(1, -1)
        for start, end in zip(pre_cumsum_idxs, cumsum_idxs)
    ]
    avg_scores = np.concatenate(scores_instance, axis=0)
    return avg_scores


def mean_descriptors_and_scores(descriptors, scores, idxs):
    cumsum_idxs = np.cumsum(idxs)
    pre_cumsum_idxs = np.cumsum(idxs)[:-1]
    pre_cumsum_idxs = np.insert(pre_cumsum_idxs, 0, 0)

    avg_descriptors = []
    for i, (start, end) in enumerate(zip(pre_cumsum_idxs, cumsum_idxs)):
        descriptors_span = descriptors[start:end]
        scores_span = scores[start:end]
        avg_descriptors.append(np.mean(descriptors_span, axis=0, keepdims=True))

    avg_descriptors = np.concatenate(avg_descriptors, axis=0)  # N*D
    avg_scores = np.ones((avg_descriptors.shape[0], 1))  # N*1 Fake score!

    return avg_descriptors, avg_scores, idxs


def get_kpt_ann(
    cfg,
    img_lists,
    feature_file_path,
    outputs_dir,
    points_idxs,
    xyzs,
    save_feature_for_each_image=True,
    use_ray=False,
    feat_3d_name_suffix="",
    verbose=True,
):
    """ Generate 3d point feature.
    @param xyzs: 3d points after filter(track length, 3d box and merge operation)
    @param points_idxs: {new_point_id: [old_point1_id, old_point2_id, ...]}.
                        new_point_id: [0, xyzs.shape[0]]
                        old_point_id*: point idx in Points3D.bin
                        This param is used to record the relationship of points after filter and before filter.
    """
    model_dir, anno_out_dir = get_default_path(cfg, outputs_dir)

    cameras, images, points3D = read_write_model.read_model(model_dir, ext=".bin")
    features = h5py.File(feature_file_path, "r")

    # step 1
    kp3d_id_mapping = id_mapping(points_idxs)
    kp3d_id_mapping = kp3d_id_mapping
    (
        feature_dim,
        kp3d_id_feature,
        kp3d_id_score,
        kp3d_idx_to_img_kp2d_idx,
    ) = count_features(img_lists, features, images, kp3d_id_mapping, verbose=verbose)

    # step 2
    if use_ray:
        # Parallel gather:
        cfg_ray = cfgs["ray"]
        if cfg_ray["slurm"]:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                num_gpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_gpus_per_worker"]),
                local_mode=cfg_ray["local_mode"],
                ignore_reinit_error=True,
            )
        pb = (
            ProgressBar(len(points_idxs), "Gather 3D annotations...")
            if verbose
            else None
        )
        points_idxs_chunked = split_dict(
            points_idxs, math.ceil(len(points_idxs) / cfg_ray["n_workers"])
        )

        kp3d_id_feature_ = ray.put(kp3d_id_feature)
        obj_refs = [
            gather_3d_ann_ray_wrapper.remote(
                kp3d_id_feature_,
                kp3d_id_score,
                xyzs,
                sub_points_ids,
                pb.actor if pb is not None else None,
                verbose=verbose,
            )
            for sub_points_ids in points_idxs_chunked
        ]
        pb.print_until_done() if pb is not None else None
        results = ray.get(obj_refs)
        filter_xyzs = np.concatenate([k for k, _, _, _ in results], axis=0)
        filter_descriptors = np.concatenate([k for _, k, _, _ in results], axis=0)
        filter_scores = np.concatenate([k for _, _, k, _ in results], axis=0)
        idxs = np.concatenate([k for _, _, _, k in results], axis=0)
        logger.info("Gather 3D annotation finish!")
    else:
        filter_xyzs, filter_descriptors, filter_scores, idxs = gather_3d_ann(
            kp3d_id_feature, kp3d_id_score, xyzs, points_idxs, verbose=verbose
        )

    (
        avg_descriptors,
        avg_scores,
        idxs,
    ) = mean_descriptors_and_scores(
        filter_descriptors,
        filter_scores,
        idxs,
    )

    anno2d_out_path = osp.join(anno_out_dir, "anno_2d.json")
    save_2d_anno(
        cfg,
        img_lists,
        features,
        filter_xyzs,
        points_idxs,
        kp3d_idx_to_img_kp2d_idx,
        anno2d_out_path,
        save_feature_for_each_img=save_feature_for_each_image,
        verbose=verbose,
    )

    avg_anno3d_out_path = osp.join(
        anno_out_dir, "anno_3d_average" + feat_3d_name_suffix + ".npz"
    )
    save_3d_anno(filter_xyzs, avg_descriptors, avg_scores, avg_anno3d_out_path)