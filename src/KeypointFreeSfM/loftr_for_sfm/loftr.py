import torch
import torch.nn as nn
from einops.einops import rearrange
import sys
sys.path.append('submodules/LoFTR/src')

from loftr.backbone import build_backbone
from loftr.utils.position_encoding import PositionEncodingSine
from loftr.loftr_module import LocalFeatureTransformer, FinePreprocess
from loftr.utils.coarse_matching import CoarseMatching
from loftr.utils.fine_matching import FineMatching

from .utils.sample_feature_from_featuremap import sample_feature_from_featuremap


class LoFTR_for_OnePose_Plus(nn.Module):
    def __init__(self, config, enable_fine_matching=True):
        super().__init__()
        # Misc
        self.config = config
        self.enable_fine_matching = enable_fine_matching

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

    def forward(self, data, **kwargs):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        feat_c0_backbone = feat_c0.clone()
        feat_c1_backbone = feat_c1.clone()
        feat_f0_backbone = feat_f0.clone()
        feat_f1_backbone = feat_f1.clone()

        if 'mkpts0_c' not in data:
            # 2. coarse-level loftr module
            # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
            feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
            feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

            mask_c0 = mask_c1 = None  # mask is useful in training
            if 'mask0' in data:
                mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
            feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

            # 3. match coarse-level
            self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        else:
            # Only fine with coarse provided
            # Convert coarse match to b_ids, i_ids, j_ids
            # NOTE: only allow batch_size == 1
            b_ids = torch.zeros(
                (data["mkpts0_c"].shape[0],), device=data["mkpts0_c"].device
            ).long()

            data['mkpts0_c'][:, 0] = torch.clip(data['mkpts0_c'][:, 0], min=0, max=data['hw0_i'][1] - 2)
            data['mkpts0_c'][:, 1] = torch.clip(data['mkpts0_c'][:, 1], min=0, max=data['hw0_i'][0] - 2)
            data['mkpts1_c'][:, 0] = torch.clip(data['mkpts1_c'][:, 0], min=0, max=data['hw1_i'][1] - 2)
            data['mkpts1_c'][:, 1] = torch.clip(data['mkpts1_c'][:, 1], min=0, max=data['hw1_i'][0] - 2)

            scale = data["hw0_i"][0] / data["hw0_c"][0]
            scale0 = (
                scale * data["scale0"][b_ids][:, [1, 0]] if "scale0" in data else scale
            )
            scale1 = (
                scale * data["scale1"][b_ids][:, [1, 0]] if "scale1" in data else scale
            )

            mkpts0_coarse_scaled = torch.round(data["mkpts0_c"] / scale0)
            mkpts1_coarse_scaled = torch.round(data["mkpts1_c"] / scale1)
            i_ids = (
                mkpts0_coarse_scaled[:, 1] * data["hw0_c"][1]
                + mkpts0_coarse_scaled[:, 0]
            ).long()
            j_ids = (
                mkpts1_coarse_scaled[:, 1] * data["hw1_c"][1]
                + mkpts1_coarse_scaled[:, 0]
            ).long()

            feat_c0, feat_c1 = None, None

            data.update(
                {"m_bids": b_ids, "b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids, 'mconf': torch.ones_like(b_ids)}
            )

        if self.enable_fine_matching:
            # 4. fine-level refinement
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
            if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

            # 5. match fine-level
            self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        else:
            data.update(
                {"mkpts0_f": data["mkpts0_c"], "mkpts1_f": data["mkpts1_c"],}
            )

        # 6. extract and return features (optional):
        if "extract_coarse_feature" in kwargs:
            if kwargs["extract_coarse_feature"]:
                feat_coarse_b_0 = sample_feature_from_featuremap(
                    feat_c0_backbone,
                    data["mkpts0_f"],
                    imghw=data["scale0"].squeeze(0)
                    * torch.tensor(data["hw0_i"]).to(data["scale0"]),
                    sample_mode="nearest",
                )
                feat_coarse_b_1 = sample_feature_from_featuremap(
                    feat_c1_backbone,
                    data["mkpts1_f"],
                    imghw=data["scale1"].squeeze(0)
                    * torch.tensor(data["hw1_i"]).to(data["scale1"]),
                    sample_mode="nearest",
                )
                data.update(
                    {
                        "feat_coarse_b_0": feat_coarse_b_0,
                        "feat_coarse_b_1": feat_coarse_b_1,
                    }
                )
        if "extract_fine_feature" in kwargs:
            if kwargs["extract_fine_feature"]:
                feat_ext0 = sample_feature_from_featuremap(
                    feat_f0_backbone,
                    data["mkpts0_f"],
                    imghw=data["scale0"].squeeze(0)
                    * torch.tensor(data["hw0_i"]).to(data["scale0"]),
                )
                feat_ext1 = sample_feature_from_featuremap(
                    feat_f1_backbone,
                    data["mkpts1_f"],
                    imghw=data["scale1"].squeeze(0)
                    * torch.tensor(data["hw1_i"]).to(data["scale1"]),
                )
                data.update({"feat_ext0": feat_ext0, "feat_ext1": feat_ext1})