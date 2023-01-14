import torch
from pytorch3d import transforms

from .residual_utils import AngleAxisRotatePoint

def depth_residual(
    depth,
    pose0,
    pose1,
    intrinsic0,
    intrinsic1,
    mkpts0_c,
    mkpts1_c,
    mkpts1_f,
    mode="geometry_error",
    confidance=None,
    **kwargs
):
    """
    Parameters:
    -------------
    pose0: torch.tensor L*6 or L*1*6
    pose1: torch.tensor L*6 or L*1*6
    depth: torch.tensor L*1 (variable) 
    intrinsic0: torch.tensor L*3*3
    intrinsic1: torch.tensor L*3*3
    mkpts0_c: L*2
    mkpts1_c: L*2
    mkpts1_f: L*2
    confidance: L*1
    """

    # Dim check
    depth = depth.squeeze(1) if len(depth.shape) == 3 else depth
    pose0 = pose0.squeeze(1) if len(pose0.shape) == 3 else pose0
    pose1 = pose1.squeeze(1) if len(pose1.shape) == 3 else pose1
    mkpts0_c = mkpts0_c.squeeze(1) if len(mkpts0_c.shape) == 3 else mkpts0_c
    mkpts1_c = mkpts1_c.squeeze(1) if len(mkpts1_c.shape) == 3 else mkpts1_c
    mkpts1_f = mkpts1_f.squeeze(1) if len(mkpts1_f.shape) == 3 else mkpts1_f

    intrinsic0 = intrinsic0.squeeze(1) if len(intrinsic0.shape) == 4 else intrinsic0
    intrinsic1 = intrinsic1.squeeze(1) if len(intrinsic1.shape) == 4 else intrinsic1

    device = depth.device

    # Unproject
    kpts0_h = (
        torch.cat([mkpts0_c, torch.ones((mkpts0_c.shape[0], 1), device=device)], dim=-1)
        * depth
    )  # (N, 3)
    kpts0_cam0 = intrinsic0.inverse() @ kpts0_h.unsqueeze(-1)  # (N*3*1)

    # Rotation and translation
    # inverse pose0
    R_inverse = transforms.so3_exponential_map(pose0[:, :3]).inverse() # (N*3*3)
    t_inverse = -1 * (R_inverse @ pose0[:, 3:6].unsqueeze(-1)).squeeze(-1) # N*3
    angle_axis_inverse = transforms.so3_log_map(R_inverse)
    pose0_inverse = torch.cat([angle_axis_inverse, t_inverse], dim=1) # N*6

    w_kpts0_world = (
        AngleAxisRotatePoint(pose0_inverse[:, :3], kpts0_cam0.squeeze(-1)) + pose0_inverse[:, 3:6]
    )  # (N*3)
    w_kpts0_cam1 = (
        AngleAxisRotatePoint(pose1[:, :3], w_kpts0_world.squeeze(-1)) + pose1[:, 3:6]
    )  # (N*3)

    # Projection
    w_kpts0_frame1_h = (intrinsic1 @ w_kpts0_cam1.unsqueeze(-1)).squeeze(-1)  # (N*3)
    w_kpts0_frame1 = w_kpts0_frame1_h[:, :2] / (w_kpts0_frame1_h[:, [2]] + 1e-4)

    if mode == "geometry_error":
        distance = w_kpts0_frame1 - mkpts1_f
    else:
        raise NotImplementedError

    if confidance is not None:
        return distance[confidance > 0], confidance
    else:
        return distance