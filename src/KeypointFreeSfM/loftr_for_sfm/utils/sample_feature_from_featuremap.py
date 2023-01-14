import torch
import torch.nn.functional as F
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange

def coord_normalization(keypoints, h, w, scale=1):
    """
    Normalize keypoints to [-1, 1] for different scales.
    Parameters:
    ---------------
    keypoints: torch.tensor N*2
        coordinates at different images scales
    """
    keypoints = keypoints - scale / 2 + 0.5  # calc down-sampled keypoints positions
    rescale_tensor = torch.tensor([(w - 1) * scale, (h - 1) * scale]).to(keypoints)
    if len(keypoints.shape) == 2:
        rescale_tensor = rescale_tensor[None]
    elif len(keypoints.shape) == 4:
        # grid scenario
        rescale_tensor = rescale_tensor[None, None, None]
    else:
        raise NotImplementedError
    keypoints /= rescale_tensor
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    return keypoints


def sample_feature_from_featuremap(
    feature_map, kpts, imghw, norm_feature=False, patch_feature_size=None, sample_mode="bilinear"
):
    """
    Sample feature from the whole feature map
    Parameters:
    -------------
    feature_map : torch.tensor C*H*W or 1*C*H*W
    kpts : torch.tensor L*2
    imghw : torch.tensor 2
        h=imghw[0], w=imghw[1]
    norm_feature : bool
        if true: return normalize feature
    patch_feature_size : int
        size of local patch around keypoints regarding to original image resolution
    """
    if len(feature_map.shape) == 4:
        # Batch size need to be 1
        assert feature_map.shape[0] == 1, "batch should to be 1"
    grid = kpts[None, :, None, :] # 1*L*1*2
    if patch_feature_size is not None:
        assert patch_feature_size>0,"invalid patch feature size!"
        # Get every point's coordinate in each grid in image resolution
        local_patch_grid = (
            create_meshgrid(
                patch_feature_size,
                patch_feature_size,
                normalized_coordinates=False,
                device=feature_map.device,
            )
            - patch_feature_size // 2
        )
        grid = grid.unsqueeze(-2) # 1*L*1*1*2
        grid = grid + local_patch_grid.long().unsqueeze(0)  # 1*L*W*W*2
        grid = rearrange(grid, "n l h w c -> n l (h w) c") # 1*L*WW*2

    # FIXME: problem here: local window is also rescaled!
    grid_n = coord_normalization(grid, imghw[0], imghw[1])

    feature = F.grid_sample(
        feature_map.unsqueeze(0) if len(feature_map.shape) == 3 else feature_map,  # 1*C*H*W
        grid_n.float(),
        mode=sample_mode,
        align_corners=True,
    )  # 1*C*L*WW or 1*C*L*1
    feature = (
        rearrange(feature, "l c h w -> l h w c").squeeze(0)
    )  # L*WW*C or L*1*C

    if patch_feature_size is not None:
        feature = rearrange(feature, "l (h w) c -> l h w c", h=patch_feature_size) # L*W*W*C
    else:
        feature = feature.squeeze(-2) # L*C

    return F.normalize(feature, p=2, dim=-1) if norm_feature else feature