import torch
import pytorch_lightning as pl
import numpy as np
import ray
from ray.actor import ActorHandle
from tqdm import tqdm

from src.KeypointFreeSfM.loftr_for_sfm import LoFTR_for_OnePose_Plus, default_cfg


def build_model(args):
    pl.seed_everything(args["seed"])

    matcher = LoFTR_for_OnePose_Plus(config=default_cfg, enable_fine_matching=True)
    # load checkpoints
    state_dict = torch.load(args["weight_path"], map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k.replace("matcher.", "")] = state_dict.pop(k)
    matcher.load_state_dict(state_dict, strict=True)
    matcher.eval()
    return matcher

def extract_results(
    data,
    matcher=None,
    extract_feature_method=None,
):
    # 1. inference
    if extract_feature_method == "fine_match_backbone":
        matcher(data, extract_coarse_feature=True, extract_fine_feature=True)
    else:
        raise NotImplementedError

    m_bids = data["m_bids"].cpu().numpy()
    assert (np.unique(m_bids) == 0).all()
    mkpts0_c = data["mkpts0_c"].cpu().numpy()
    mkpts1_c = data["mkpts1_c"].cpu().numpy()
    mkpts0_f = data["mkpts0_f"].cpu().numpy()
    mkpts1_f = data["mkpts1_f"].cpu().numpy()
    mkpts0_idx = data["mkpts0_idx"].cpu().numpy()  # NOTE: from original data, just passby
    scale0 = data["scale0"].cpu().numpy()
    scale1 = data["scale1"].cpu().numpy()

    if extract_feature_method == "fine_match_backbone":
        feature0 = data["feat_ext0"].cpu().numpy()
        feature1 = data["feat_ext1"].cpu().numpy()
    else:
        raise NotImplementedError

    if "feat_coarse_b_0" in data:
        feature_c0 = data['feat_coarse_b_0'].cpu().numpy()
        feature_c1 = data['feat_coarse_b_1'].cpu().numpy()
    else:
        raise NotImplementedError
    
    return (
        mkpts0_c,
        mkpts1_c,
        mkpts0_f,
        mkpts1_f,
        mkpts0_idx,
        scale0,
        scale1,
        feature_c0,
        feature_c1,
        feature0,
        feature1,
    )


# Used for two view match and pose & depth refinement
@torch.no_grad()
def matchWorker(
    dataset,
    subset_ids,
    matcher,
    extract_feature_method=None,
    pba: ActorHandle = None,
    verbose=True,
):
    """extract matches from part of the possible image pair permutations"""
    matcher.cuda()
    results_dict = {}

    if verbose:
        subset_ids = tqdm(subset_ids) if pba is None else subset_ids
    else:
        assert pba is None
        subset_ids = subset_ids

    for subset_id in subset_ids:
        data = dataset[subset_id]
        frameID0, frameID1 = data["frame0_colmap_id"], data["frame1_colmap_id"]
        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }

        (
            mkpts0_c,
            mkpts1_c,
            mkpts0_f,
            mkpts1_f,
            mkpts0_idx,
            scale0,
            scale1,
            feature_c0,
            feature_c1,
            feature0,
            feature1,
        ) = extract_results(
            data_c,
            matcher=matcher,
            extract_feature_method=extract_feature_method,
        )

        # 3. extract results
        pair_name = "-".join([str(frameID0), str(frameID1)])
        results_dict[pair_name] = {  # colmap frame id
            "mkpts0_c": mkpts0_c,  # N*2
            "mkpts1_c": mkpts1_c,
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "mkpts0_idx": mkpts0_idx,
            "scale0": scale0,  # 1*2
            "scale1": scale1,
            "feature_c0": feature_c0,
            "feature_c1": feature_c1,
            "feature0": feature0,
            "feature1": feature1,
        }
        if pba is not None:
            pba.update.remote(1)

    return results_dict


@ray.remote(num_cpus=1, num_gpus=0.25, max_calls=1)  # release gpu after finishing
def matchWorker_ray_wrapper(*args, **kwargs):
    return matchWorker(*args, **kwargs)
