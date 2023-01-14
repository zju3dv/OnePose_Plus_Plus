import os
import ray
import math
import h5py
import os.path as osp
from loguru import logger
from typing import ChainMap

from src.utils.ray_utils import ProgressBar, chunks, chunk_index, split_dict
from src.utils.data_io import save_h5, load_h5
from .utils import Match2Pts2D
from .coarse_match_worker import *
from ..dataset.loftr_coarse_dataset import LoftrCoarseDataset

cfgs = {
    "data": {"img_resize": 512, "df": 8, "shuffle": True},
    "matcher": {
        "model": {
            "weight_path": "weight/LoFTR_wsize9.ckpt",
            "seed": 666,
        },
        "pair_name_split": " ",
    },
    "coarse_match_debug": True,
    "ray": {
        "slurm": False,
        "n_workers": 4, 
        "n_cpus_per_worker": 1,
        "n_gpus_per_worker": 0.25,
        "local_mode": False,
    },
}


def detector_free_coarse_matching(
    image_lists,
    covis_pairs_out,
    feature_out,
    match_out,
    use_ray=False,
    verbose=False,
):
    # Build dataset:
    dataset = LoftrCoarseDataset(cfgs["data"], image_lists, covis_pairs_out)

    # Construct directory
    base_dir = feature_out.rsplit("/", 1)[0]
    os.makedirs(base_dir, exist_ok=True)
    cache_dir = osp.join(feature_out.rsplit("/", 1)[0], "raw_matches.h5")

    if use_ray:
        # Initial ray:
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

        # Matcher runner
        if not cfgs["coarse_match_debug"] and osp.exists(cache_dir):
            matches = load_h5(cache_dir, transform_slash=True)
            logger.info("Caches raw matches loaded!")
        else:
            pb = (
                ProgressBar(len(dataset), "Matching image pairs...")
                if verbose
                else None
            )
            all_subset_ids = chunk_index(
                len(dataset), math.ceil(len(dataset) / cfg_ray["n_workers"])
            )
            obj_refs = [
                match_worker_ray_wrapper.remote(
                    dataset,
                    subset_ids,
                    cfgs["matcher"],
                    pb.actor if pb is not None else None,
                    verbose=verbose,
                )
                for subset_ids in all_subset_ids
            ]
            pb.print_until_done() if pb is not None else None
            results = ray.get(obj_refs)
            matches = dict(ChainMap(*results))
            logger.info("Matcher finish!")

            save_h5(matches, cache_dir)
            logger.info(f"Raw matches cached: {cache_dir}")

        # Combine keypoints
        n_imgs = len(dataset.img_dir)
        pb = ProgressBar(n_imgs, "Combine points 2D...") if verbose else None
        all_kpts = Match2Pts2D(
            matches, dataset.img_dir, name_split=cfgs["matcher"]["pair_name_split"]
        )
        sub_kpts = chunks(all_kpts, math.ceil(n_imgs / cfg_ray["n_workers"]))
        obj_refs = [
            points2D_worker_ray_wrapper.remote(sub_kpt, pb.actor if pb is not None else None, verbose)
            for sub_kpt in sub_kpts
        ]
        pb.print_until_done() if pb is not None else None
        keypoints = dict(ChainMap(*ray.get(obj_refs)))
        logger.info("Combine keypoints finish!")

        # Convert keypoints match to keypoints indexs
        pb = ProgressBar(len(matches), "Updating matches...") if verbose else None
        _keypoints_ref = ray.put(keypoints)
        obj_refs = [
            update_matches_ray_wrapper.remote(
                sub_matches,
                _keypoints_ref,
                pb.actor if pb is not None else None,
                verbose=verbose,
                pair_name_split=cfgs["matcher"]["pair_name_split"],
            )
            for sub_matches in split_dict(
                matches, math.ceil(len(matches) / cfg_ray["n_workers"])
            )
        ]
        pb.print_until_done() if pb is not None else None
        updated_matches = dict(ChainMap(*ray.get(obj_refs)))

        # Post process keypoints:
        keypoints = {k: v for k, v in keypoints.items() if isinstance(v, dict)}
        pb = ProgressBar(len(keypoints), "Post-processing points 2D...") if verbose else None
        obj_refs = [
            transform_points2D_ray_wrapper.remote(sub_kpts, pb.actor if pb is not None else None, verbose)
            for sub_kpts in split_dict(
                keypoints, math.ceil(len(keypoints) / cfg_ray["n_workers"])
            )
        ]
        pb.print_until_done() if pb is not None else None
        kpts_scores = ray.get(obj_refs)
        final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
        final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))
    else:
        # Matcher runner
        if not cfgs["coarse_match_debug"] and osp.exists(cache_dir):
            matches = load_h5(cache_dir, transform_slash=True)
            logger.info("Caches raw matches loaded!")
        else:
            all_ids = np.arange(0, len(dataset))
            matches = match_worker(dataset, all_ids, cfgs["matcher"], verbose=verbose)
            logger.info("Matcher finish!")

            # over write anyway
            save_h5(matches, cache_dir)
            logger.info(f"Raw matches cached: {cache_dir}")

        # Combine keypoints
        n_imgs = len(dataset.img_dir)
        logger.info("Combine 2D points!")
        all_kpts = Match2Pts2D(
            matches, dataset.img_dir, name_split=cfgs["matcher"]["pair_name_split"]
        )
        sub_kpts = chunks(all_kpts, math.ceil(n_imgs / 1))  # equal to only 1 worker
        obj_refs = [points2D_worker(sub_kpt, verbose=verbose) for sub_kpt in sub_kpts]
        keypoints = dict(ChainMap(*obj_refs))

        # Convert points2D match to points2D indexs
        logger.info("Update matches")
        obj_refs = [
            update_matches(
                sub_matches,
                keypoints,
                verbose=verbose,
                pair_name_split=cfgs["matcher"]["pair_name_split"],
            )
            for sub_matches in split_dict(matches, math.ceil(len(matches) / 1))
        ]
        updated_matches = dict(ChainMap(*obj_refs))

        # Post process points2D:
        keypoints = {k: v for k, v in keypoints.items() if isinstance(v, dict)}
        logger.info("Post-processing keypoints...")
        kpts_scores = [
            transform_points2D(sub_kpts, verbose=verbose)
            for sub_kpts in split_dict(keypoints, math.ceil(len(keypoints) / 1))
        ]
        final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
        final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))

    # Save 2D points coords:
    with h5py.File(feature_out, "w") as feature_file:
        for image_name, keypoints in final_keypoints.items():
            grp = feature_file.create_group(image_name)
            grp.create_dataset("keypoints", data=keypoints)

            # Fake features:
            dim = 256
            descriptors = np.zeros((dim, keypoints.shape[0]))
            grp.create_dataset("descriptors", data=descriptors)

            # Fake scores:
            scores = np.ones((keypoints.shape[0],))
            grp.create_dataset("scores", data=scores)

    # Save matches:
    with h5py.File(match_out, "w") as match_file:
        for pair_name, matches in updated_matches.items():
            name0, name1 = pair_name.split(cfgs["matcher"]["pair_name_split"])
            pair = names_to_pair(name0, name1)

            grp = match_file.create_group(pair)
            grp.create_dataset("matches", data=matches)
            grp.create_dataset("matching_scores", data=np.ones((matches.shape[0],))) # Place holder

            grp.create_dataset("matches0", data=matches)

    return final_keypoints, updated_matches
