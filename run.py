from copy import deepcopy
import os
import os.path as osp
import natsort

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
import hydra
import math
import ray

import os.path as osp
from tqdm import tqdm

from loguru import logger
from pathlib import Path
from omegaconf import DictConfig


from src.utils.ray_utils import ProgressBar, chunks

def sfm(cfg):
    """ Sparse reconstruction and postprocess (on 3d points and features)"""
    data_dirs = cfg.dataset.data_dir

    if isinstance(data_dirs, str):
        # Parse all objects in the directory
        num_seq = cfg.dataset.num_seq
        exception_obj_name_list = cfg.dataset.exception_obj_names
        top_k_obj = cfg.dataset.top_k_obj
        if num_seq is not None:
            assert num_seq > 0
        logger.info(
            f"Process all objects in directory:{data_dirs}, process: {num_seq if num_seq is not None else 'all'} sequences"
        )

        object_names = os.listdir(data_dirs)[:top_k_obj]
        data_dirs_list = []

        if cfg.dataset.ids is not None:
            # Use data ids:
            id2full_name = {name[:4]: name for name in object_names if "-" in name}
            object_names = [
                id2full_name[id] for id in cfg.dataset.ids if id in id2full_name
            ]

        for object_name in object_names:
            if "-" not in object_name:
                continue

            if object_name in exception_obj_name_list:
                continue
            sequence_names = sorted(os.listdir(osp.join(data_dirs, object_name)))
            sequence_names = [
                sequence_name
                for sequence_name in sequence_names
                if "-" in sequence_name
            ][:num_seq]
            data_dirs_list.append(
                " ".join([osp.join(data_dirs, object_name)] + sequence_names)
            )

        data_dirs = data_dirs_list

    if not cfg.use_global_ray:
        sfm_worker(data_dirs, cfg)
    else:
        # Init ray
        if cfg.ray.slurm:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(cfg.ray.n_workers * cfg.ray.n_cpus_per_worker),
                num_gpus=math.ceil(cfg.ray.n_workers * cfg.ray.n_gpus_per_worker),
                local_mode=cfg.ray.local_mode,
                ignore_reinit_error=True,
            )
        logger.info(f"Use ray for SfM mapping, total: {cfg.ray.n_workers} workers")

        pb = ProgressBar(len(data_dirs), "SfM Mapping begin...")
        all_subsets = chunks(data_dirs, math.ceil(len(data_dirs) / cfg.ray.n_workers))
        sfm_worker_results = [
            sfm_worker_ray_wrapper.remote(
                subset_data_dirs, cfg, worker_id=id, pba=pb.actor
            )
            for id, subset_data_dirs in enumerate(all_subsets)
        ]
        pb.print_until_done()
        results = ray.get(sfm_worker_results)


def sfm_worker(data_dirs, cfg, worker_id=0, pba=None):
    logger.info(
        f"Worker: {worker_id} will process: {[(data_dir.split(' ')[0]).split('/')[-1][:4] for data_dir in data_dirs]}, total: {len(data_dirs)} objects"
    )
    data_dirs = tqdm(data_dirs) if pba is None else data_dirs
    for data_dir in data_dirs:
        logger.info(f"Processing {data_dir}.")
        root_dir, sub_dirs = data_dir.split(" ")[0], data_dir.split(" ")[1:]

        img_lists = []
        ext_bag = [".png", ".jpg"]
        for sub_dir in sub_dirs:
            seq_dir = osp.join(root_dir, sub_dir)
            img_dir_name = 'color'
            img_name_lists = os.listdir(osp.join(seq_dir, img_dir_name))
            img_lists += [
                osp.join(seq_dir, img_dir_name, img_name)
                for img_name in img_name_lists
                if osp.splitext(img_name)[1] in ext_bag
            ]

        # Extract subset images:
        down_img_lists = []
        down_ratio = cfg.sfm.down_ratio
        for id, img_file in enumerate(natsort.natsorted(img_lists)):
            if id % down_ratio == 0:
                down_img_lists.append(img_file)
        img_lists = down_img_lists

        if len(img_lists) == 0:
            logger.info(f"No png image in {root_dir}")
            if pba is not None:
                pba.update.remote(1)
            continue

        obj_name = root_dir.split("/")[-1]
        outputs_dir_root = cfg.dataset.outputs_dir

        sfm_core(cfg, img_lists, outputs_dir_root, obj_name)
        postprocess(cfg, img_lists, root_dir, sub_dirs, outputs_dir_root, obj_name)

        logger.info(f"Finish Processing {data_dir}.")
        if pba is not None:
            pba.update.remote(1)
    logger.info(f"Worker{worker_id} finish!")
    return None


@ray.remote
def sfm_worker_ray_wrapper(*args, **kwargs):
    return sfm_worker(*args, **kwargs)


def sfm_core(cfg, img_lists, outputs_dir_root, obj_name):
    """ 
    Keypoint-Free SfM: coarse reconstruction (including match features, triangulation), post optimization
    """
    from src.sfm_utils import (
        generate_empty,
        triangulation,
        pairs_exhaustive_all, pairs_from_index, pairs_from_poses
    )
    from src.KeypointFreeSfM import coarse_match, post_optimization

    outputs_dir = osp.join(
        outputs_dir_root,
        "outputs_"
        + cfg.match_type
        + "_"
        + cfg.network.detection
        + "_"
        + cfg.network.matching,
        obj_name,
    )
    vis3d_pth = osp.join(
        outputs_dir_root,
        "outputs_"
        + cfg.match_type
        + "_"
        + cfg.network.detection
        + "_"
        + cfg.network.matching,
        "vis3d",
        obj_name,
    )

    feature_out = osp.join(outputs_dir, f"feats-{cfg.network.detection}.h5")
    covis_num = cfg.sfm.covis_num
    covis_pairs_out = osp.join(outputs_dir, f"pairs-covis{covis_num}.txt")
    matches_out = osp.join(outputs_dir, f"matches-{cfg.network.matching}.h5")
    empty_dir = osp.join(outputs_dir, "sfm_empty")
    deep_sfm_dir = osp.join(outputs_dir, "sfm_ws")

    if cfg.overwrite_all:
        os.system(f"rm -rf {outputs_dir}")
        os.system(f"rm -rf {vis3d_pth}")
    Path(outputs_dir).mkdir(exist_ok=True, parents=True)

    if (
        not osp.exists(osp.join(deep_sfm_dir, "model_coarse"))
        or cfg.overwrite_coarse
    ):
        logger.info("Keypoint-Free SfM coarse reconstruction begin...")
        os.system(f"rm -rf {empty_dir}")
        os.system(f"rm -rf {deep_sfm_dir}")
        os.system(
            f'rm -rf {osp.join(covis_pairs_out.rsplit("/", 1)[0], "fine_matches.pkl")}'
        )  # Force refinement to recompute fine match

        if covis_num == -1:
            pairs_exhaustive_all.exhaustive_all_pairs(
                img_lists, covis_pairs_out
            )
        else:
            if cfg.sfm.gen_cov_from == 'index':
                pairs_from_index.covis_from_index(
                    img_lists,
                    covis_pairs_out,
                    num_matched=covis_num,
                    gap=cfg.sfm.gap,
                )
            elif cfg.sfm.gen_cov_from == 'pose':
                pairs_from_poses.covis_from_pose(
                    img_lists,
                    covis_pairs_out,
                    covis_num,
                    min_rotation=cfg.sfm.min_rotation
                )
            else:
                raise NotImplementedError

        coarse_match.detector_free_coarse_matching(
            img_lists,
            covis_pairs_out,
            feature_out,
            matches_out,
            use_ray=cfg.use_local_ray,
            verbose=cfg.verbose
        )
        generate_empty.generate_model(img_lists, empty_dir)

        if cfg.use_global_ray:
            # Need to ask for gpus!
            triangulation_results = triangulation.main_ray_wrapper.remote(
                deep_sfm_dir,
                empty_dir,
                outputs_dir,
                covis_pairs_out,
                feature_out,
                matches_out,
                match_model=cfg.network.matching,
                image_dir=None,
                verbose=cfg.verbose,
            )
            results = ray.get(triangulation_results)
        else:
            triangulation.main(
                deep_sfm_dir,
                empty_dir,
                outputs_dir,
                covis_pairs_out,
                feature_out,
                matches_out,
                match_model=cfg.network.matching,
                image_dir=None,
                verbose=cfg.verbose,
            )

            os.system(
                f"mv {feature_out} {osp.splitext(feature_out)[0] + '_coarse' + osp.splitext(feature_out)[1]}"
            )
            if cfg.enable_post_refine:
                assert osp.exists(osp.join(deep_sfm_dir, "model"))
                os.system(
                    f"mv {osp.join(deep_sfm_dir, 'model')} {osp.join(deep_sfm_dir, 'model_coarse')}"
                )

    if cfg.enable_post_refine:
        if (
            not osp.exists(osp.join(deep_sfm_dir, "model"))
            or cfg.overwrite_fine
        ):
            assert osp.exists(
                osp.join(deep_sfm_dir, "model_coarse")
            ), f"model_coarse not exist under: {deep_sfm_dir}, please set 'cfg.overwrite_coarse = True'"
            os.system(f"rm -rf {osp.join(deep_sfm_dir, 'model')}")

            # configs for post optimization:
            post_optim_configs = cfg.post_optim if 'post_optim' in cfg else None

            logger.info("Keypoint-Free SfM post refinement begin...")
            state = post_optimization.post_optimization(
                img_lists,
                covis_pairs_out,
                colmap_coarse_dir=osp.join(deep_sfm_dir, "model_coarse"),
                refined_model_save_dir=osp.join(deep_sfm_dir, "model"),
                match_out_pth=matches_out,
                feature_out_pth=feature_out,
                use_global_ray=cfg.use_global_ray,
                fine_match_use_ray=cfg.use_local_ray,
                vis3d_pth=vis3d_pth,
                verbose=cfg.verbose,
                args=post_optim_configs
            )
            if state == False:
                logger.error("Coarse reconstruction failed!")
    else:
        raise NotImplementedError

def postprocess(cfg, img_lists, root_dir, sub_dirs, outputs_dir_root, obj_name):
    """ Filter points and average feature"""
    from src.sfm_utils.postprocess import filter_points, feature_process, filter_tkl

    bbox_path = osp.join(root_dir, "box3d_corners.txt")
    trans_box_path = None

    outputs_dir = osp.join(
        outputs_dir_root,
        "outputs_"
        + cfg.match_type
        + "_"
        + cfg.network.detection
        + "_"
        + cfg.network.matching,
        obj_name,
    )
    vis3d_pth = osp.join(
        outputs_dir_root,
        "outputs_"
        + cfg.match_type
        + "_"
        + cfg.network.detection
        + "_"
        + cfg.network.matching,
        "vis3d",
        obj_name,
    )
    feature_out = osp.join(outputs_dir, f"feats-{cfg.network.detection}.h5")
    deep_sfm_dir = osp.join(outputs_dir, "sfm_ws")
    model_path = osp.join(deep_sfm_dir, "model")

    model_filted_bbox_path = osp.join(deep_sfm_dir, "model_filted_bbox")
    os.makedirs(model_filted_bbox_path, exist_ok=True)
    if not cfg.post_process.skip_bbox_filter:
        filter_points.filter_bbox(
            model_path,
            model_filted_bbox_path,
            bbox_path,
            box_trans_path=trans_box_path,
        )  # crop 3d points by 3d box and save as colmap format
    else:
        os.system(f"rm -rf {model_filted_bbox_path}")
        os.system(
            f"cp -r {model_path} {model_filted_bbox_path}"
        )

    # select track length to limit the number of 3d points below thres.
    track_length, points_count_list = filter_tkl.get_tkl(
        model_filted_bbox_path, thres=cfg.dataset.max_num_kp3d, show=False
    )
    filter_tkl.vis_tkl_filtered_pcds(
        model_filted_bbox_path,
        points_count_list,
        track_length,
        outputs_dir,
        vis3d_pth,
    )  # visualization only

    xyzs, points_ids = filter_points.filter_track_length(
        model_filted_bbox_path, track_length
    )  # crop 3d points by 3d box and track length

    merge_xyzs, merge_idxs = filter_points.merge(xyzs, points_ids)  # merge 3d points by distance between points

    # Save loftr coarse keypoints:
    cfg_coarse = deepcopy(cfg)
    cfg_coarse.network.detection = "loftr_coarse"
    feature_coarse_path = (
        osp.splitext(feature_out)[0] + "_coarse" + osp.splitext(feature_out)[1]
    )
    feature_process.get_kpt_ann(
        cfg_coarse,
        img_lists,
        feature_coarse_path,
        outputs_dir,
        merge_idxs,
        merge_xyzs,
        save_feature_for_each_image=False,
        feat_3d_name_suffix="_coarse",
        use_ray=cfg.use_local_ray,
        verbose=cfg.verbose,
    )

    # Save fine level points and features:
    feature_process.get_kpt_ann(
        cfg,
        img_lists,
        feature_out,
        outputs_dir,
        merge_idxs,
        merge_xyzs,
        save_feature_for_each_image=False,
        use_ray=cfg.use_local_ray,
        verbose=cfg.verbose,
    )


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
