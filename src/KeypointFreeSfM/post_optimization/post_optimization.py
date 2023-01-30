import os
import os.path as osp

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
from loguru import logger
import ray
from src.utils.data_io import load_obj, save_obj

from ..dataset.coarse_colmap_dataset import CoarseReconDataset
from .data_construct import (
    MatchingPairData,
    ConstructOptimizationData,
)
from .matcher_model import *
from .optimizer.optimizer import Optimizer
from .feature_aggregation import feature_aggregation_and_update

cfgs = {
    "coarse_recon_data": {
        "img_resize": None, # None means use original image size
        "df": 8,
        "feature_track_assignment_strategy": "greedy",
        "verbose": False,
    },
    "fine_match_debug": True,
    "fine_matcher": {
        "model": {
            "weight_path": "weight/LoFTR_wsize9.ckpt",
            "seed": 666,
        },
        "extract_feature_method": "fine_match_backbone",
        "ray": {
            "slurm": False,
            "n_workers": 4,
            "n_cpus_per_worker": 1,
            "n_gpus_per_worker": 0.25,
            "local_mode": False,
        },
    },
    "optimizer": {
        # Dataloading related:
        "num_workers": 12,
        "batch_size": 2000,
        "solver_type": "FirstOrder",
        "residual_mode": "geometry_error",  
        "optimize_lr": {
            "depth": 3e-2,
        },  # Only available for first order solver
        "optim_procedure": ["depth"],
        "image_i_f_scale": 2,
        "verbose": False,
    },
    "feature_aggregation_method": "avg",
    "visualize": True,  # vis3d visualize
    "evaluation": False,
}

def post_optimization(
    image_lists,
    covis_pairs_pth,
    colmap_coarse_dir,
    refined_model_save_dir,
    match_out_pth,
    feature_out_pth=None,  # Used to update feature
    use_global_ray=False,
    fine_match_use_ray=False,  # Use ray for fine match
    pre_sfm=False,
    vis3d_pth=None,
    verbose=True,
    args=None,
):
    # Overwrite some configs
    cfgs["coarse_recon_data"]["verbose"] = verbose
    cfgs["optimizer"]["verbose"] = verbose
    if args is not None:
        cfgs["coarse_recon_data"]["feature_track_assignment_strategy"] = args[
            "coarse_recon_data"
        ]["feature_track_assignment_strategy"]
        cfgs["optimizer"]["residual_mode"] = args["optimizer"]["residual_mode"]
        cfgs["optimizer"]["optimize_lr"]['depth'] = args["optimizer"]["optimize_lr"]['depth']

        if 'solver_type' in args['optimizer']:
            cfgs['optimizer']['solver_type'] = args['optimizer']['solver_type']

    # Construct scene data
    coarse_recon_dataset = CoarseReconDataset(
        cfgs["coarse_recon_data"],
        image_lists,
        covis_pairs_pth,
        colmap_coarse_dir,
        refined_model_save_dir,
        pre_sfm=pre_sfm,
        vis_path=vis3d_pth if vis3d_pth is not None else None,
    )
    logger.info("Scene data construct finish!")

    state = coarse_recon_dataset.state
    if state == False:
        logger.warning(
            f"Failed to build coarse reconstruction dataset! Coarse reconstructed point3D or images or cameras is empty!"
        )
        return state

    # Construct matching data
    matching_pairs_dataset = MatchingPairData(coarse_recon_dataset)

    # 2D point refinement:
    save_path = osp.join(match_out_pth.rsplit("/", 2)[0], "fine_matches.pkl")
    if not osp.exists(save_path) or cfgs["fine_match_debug"]:
        logger.info(f"2D points refinement begin!")
        fine_match_results_dict = fine_matcher(
            cfgs["fine_matcher"],
            matching_pairs_dataset,
            use_ray=fine_match_use_ray,
            verbose=verbose,
        )
        save_obj(fine_match_results_dict, save_path)
    else:
        logger.info(f"Fine matches exists! Load from {save_path}")
        fine_match_results_dict = load_obj(save_path)

    # Construct depth optimization data:
    optimization_data = ConstructOptimizationData(
        coarse_recon_dataset, fine_match_results_dict
    )

    # Post optimization:
    optimizer = Optimizer(optimization_data, cfgs["optimizer"])
    if use_global_ray:
        # Need to ask for gpus
        optimize_results = optimizer.start_optimize_ray_wrapper.remote(optimizer)
        results_dict = ray.get(optimize_results)
    else:
        results_dict = optimizer.start_optimize()

    # Update point cloud after refine:
    coarse_recon_dataset.update_optimize_results_to_colmap(
        results_dict, visualize=cfgs["visualize"]
    )

    # Update feature
    if feature_out_pth is not None:
        feature_aggregation_and_update(
            coarse_recon_dataset,
            fine_match_results_dict,
            feature_out_pth=feature_out_pth,
            image_lists=image_lists,
            aggregation_method=cfgs["feature_aggregation_method"],
            verbose=verbose,
        )

    return state