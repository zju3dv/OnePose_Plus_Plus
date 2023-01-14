import ray
import os

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm

from .utils import agg_groupby_2d
from ..loftr_for_sfm import LoFTR_for_OnePose_Plus, default_cfg

def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))

def build_model(args):
    pl.seed_everything(args['seed'])

    matcher = LoFTR_for_OnePose_Plus(config=default_cfg, enable_fine_matching=False)
    # load checkpoints
    state_dict = torch.load(args['weight_path'], map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k.replace("matcher.", "")] = state_dict.pop(k)
    matcher.load_state_dict(state_dict, strict=True)
    matcher.eval()

    return matcher

@torch.no_grad()
def extract_matches(data, matcher=None):
    # 1. inference
    matcher(data)

    """extract predictions assuming bs==1"""
    m_bids = data["m_bids"].cpu().numpy()
    assert (np.unique(m_bids) == 0).all()
    mkpts0 = data["mkpts0_f"].cpu().numpy() # N*2
    mkpts1 = data["mkpts1_f"].cpu().numpy() # N*2
    mconfs = data["mconf"].cpu().numpy() # N

    return mkpts0, mkpts1, mconfs

@torch.no_grad()
def match_worker(dataset, subset_ids, args, pba=None, verbose=True):
    """extract matches from part of the possible image pair permutations"""
    matcher = build_model(args['model'])
    matcher.cuda()
    matches = {}

    if verbose:
        subset_ids = tqdm(subset_ids) if pba is None else subset_ids
    else:
        assert pba is None
        subset_ids = subset_ids

    # match all permutations
    for id, subset_id in enumerate(subset_ids):
        data = dataset[subset_id]
        f_name0, f_name1 = data['pair_key']
        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        mkpts0, mkpts1, mconfs = extract_matches(
            data_c,
            matcher=matcher,
        )

        # Extract matches (kpts-pairs & scores)
        matches[args['pair_name_split'].join([f_name0, f_name1])] = np.concatenate(
            [mkpts0, mkpts1, mconfs[:, None]], -1
        )  # (N, 5)

        if pba is not None:
            pba.update.remote(1)
    return matches 

@ray.remote(num_cpus=1, num_gpus=0.25, max_calls=1)  # release gpu after finishing
def match_worker_ray_wrapper(*args, **kwargs):
    return match_worker(*args, **kwargs)

def points2D_worker(name_kpts, pba=None, verbose=True):
    """merge 2D points associated with one image.
    python >= 3.7 only.
    """
    keypoints = {}
    if verbose:
        name_kpts = tqdm(name_kpts) if pba is None else name_kpts
    else:
        assert pba is None
        name_kpts = name_kpts

    for name, kpts in name_kpts:
        # filtering
        kpt2score = agg_groupby_2d(kpts[:, :2].astype(int), kpts[:, -1], agg="sum")
        kpt2id_score = {
            k: (i, v)
            for i, (k, v) in enumerate(
                sorted(kpt2score.items(), key=lambda kv: kv[1], reverse=True)
            )
        }
        keypoints[name] = kpt2id_score

        if pba is not None:
            pba.update.remote(1)
    return keypoints

@ray.remote(num_cpus=1)
def points2D_worker_ray_wrapper(*args, **kwargs):
    return points2D_worker(*args, **kwargs)

def update_matches(matches, keypoints, pba=None, verbose=True, **kwargs):
    # convert match to indices
    ret_matches = {}

    if verbose:
        matches_items = tqdm(matches.items()) if pba is None else matches.items()
    else:
        assert pba is None
        matches_items = matches.items()

    for k, v in matches_items:
        mkpts0, mkpts1 = (
            map(tuple, v[:, :2].astype(int)),
            map(tuple, v[:, 2:4].astype(int)),
        )
        name0, name1 = k.split(kwargs['pair_name_split'])
        _kpts0, _kpts1 = keypoints[name0], keypoints[name1]

        mids = np.array(
            [
                [_kpts0[p0][0], _kpts1[p1][0]]
                for p0, p1 in zip(mkpts0, mkpts1)
                if p0 in _kpts0 and p1 in _kpts1
            ]
        )

        assert (
            len(mids) == v.shape[0]
        ), f"len mids: {len(mids)}, num matches: {v.shape[0]}"
        if len(mids) == 0:
            mids = np.empty((0, 2))

        ret_matches[k] = mids.astype(int)  # (N,2)
        if pba is not None:
            pba.update.remote(1)

    return ret_matches

@ray.remote(num_cpus=1)
def update_matches_ray_wrapper(*args, **kwargs):
    return update_matches(*args, **kwargs)


def transform_points2D(keypoints, pba=None, verbose=True):
    """assume points2D sorted w.r.t. score"""
    ret_kpts = {}
    ret_scores = {}

    if verbose:
        keypoints_items = tqdm(keypoints.items()) if pba is None else keypoints.items()
    else:
        assert pba is None
        keypoints_items = keypoints.items()

    for k, v in keypoints_items:
        v = {_k: _v for _k, _v in v.items() if len(_k) == 2}
        kpts = np.array([list(kpt) for kpt in v.keys()]).astype(np.float32)
        scores = np.array([s[-1] for s in v.values()]).astype(np.float32)
        assert len(kpts) != 0, "corner-case n_kpts=0 not handled."
        ret_kpts[k] = kpts
        ret_scores[k] = scores
        if pba is not None:
            pba.update.remote(1)
    return ret_kpts, ret_scores

@ray.remote(num_cpus=1)
def transform_points2D_ray_wrapper(*args, **kwargs):
    return transform_points2D(*args, **kwargs)
