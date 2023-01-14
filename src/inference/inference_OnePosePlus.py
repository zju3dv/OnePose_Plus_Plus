from itertools import chain
import ray
import os
import math
import numpy as np
from loguru import logger
import torch

from src.datasets.OnePosePlus_inference_dataset import OnePosePlusInferenceDataset
from src.utils.ray_utils import ProgressBar, chunks, chunk_index, split_dict
from src.models.OnePosePlus.OnePosePlusModel import OnePosePlus_model
from src.utils.metric_utils import aggregate_metrics

from .inference_OnePosePlus_worker import (
    inference_onepose_plus_worker, inference_onepose_plus_worker_ray_wrapper
)

args = {
    "ray": {
        "slurm": False,
        "n_workers": 2,
        "n_cpus_per_worker": 1,
        "n_gpus_per_worker": 0.5,
        "local_mode": False,
    },
}

def build_model(model_configs, ckpt_path):
    match_model = OnePosePlus_model(model_configs)
    # load checkpoints
    logger.info(f"Load ckpt:{ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k.replace("matcher.", "")] = state_dict.pop(k)

    match_model.load_state_dict(state_dict, strict=True)
    match_model.eval()
    return match_model

def inference_onepose_plus(
    sfm_results_dir, all_image_paths, cfg, use_ray=True, verbose=True
):
    """
    Inference for one object
    """
    # Build dataset:
    dataset = OnePosePlusInferenceDataset(
        sfm_results_dir,
        all_image_paths,
        load_3d_coarse=cfg.datamodule.load_3d_coarse,
        shape3d=cfg.datamodule.shape3d_val,
        img_pad=cfg.datamodule.img_pad,
        img_resize=cfg.datamodule.img_resize,
        df=cfg.datamodule.df,
        pad=cfg.datamodule.pad3D,
        load_pose_gt=True,
        n_images=None
    )
    match_model = build_model(cfg['model']["OnePosePlus"], cfg['model']['pretrained_ckpt'])

    # Run matching
    if use_ray:
        # Initial ray:
        cfg_ray = args["ray"]
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
            ProgressBar(len(dataset), "Matching image pairs...")
            if verbose
            else None
        )
        all_subset_ids = chunk_index(
            len(dataset), math.ceil(len(dataset) / cfg_ray["n_workers"])
        )
        all_subset_ids = all_subset_ids

        obj_refs = [
            inference_onepose_plus_worker_ray_wrapper.remote(
                dataset,
                match_model,
                subset_ids,
                cfg['model'],
                pb.actor if pb is not None else None,
                verbose=verbose,
            )
            for subset_ids in all_subset_ids
        ]
        pb.print_until_done() if pb is not None else None
        results = ray.get(obj_refs)

        results = list(chain(*results))
        logger.info("Matcher finish!")
    else:
        all_ids = np.arange(0, len(dataset))
        results = inference_onepose_plus_worker(dataset, match_model, all_ids, cfg['model'], verbose=verbose)
        logger.info("Match and compute pose error finish!")
    
    # Parse results:
    R_errs = []
    t_errs = []
    if 'ADD_metric' in results[0]:
        add_metric = []
        proj2d_metric = []
    else:
        add_metric = None
        proj2d_metric = None
    
    # Gather results metrics:
    for result in results:
        R_errs.append(result['R_errs'])
        t_errs.append(result['t_errs'])
        if add_metric is not None:
            add_metric.append(result['ADD_metric'])
            proj2d_metric.append(result['proj2D_metric'])
    
    # Aggregate metrics: 
    pose_errs = {'R_errs': R_errs, "t_errs": t_errs}
    if add_metric is not None:
        pose_errs.update({'ADD_metric': add_metric, "proj2D_metric": proj2d_metric})
    metrics = aggregate_metrics(pose_errs, cfg['model']['eval_metrics']['pose_thresholds'])

    return metrics