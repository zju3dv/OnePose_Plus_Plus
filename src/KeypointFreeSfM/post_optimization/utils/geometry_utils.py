import numpy as np
import torch
from pytorch3d import transforms
from src.utils.colmap.read_write_model import qvec2rotmat


def convert_pose2T(pose):
    # pose: [R: 3*3, t: 3]
    R, t = pose
    return np.concatenate(
        [np.concatenate([R, t[:, None]], axis=1), [[0, 0, 0, 1]]], axis=0
    )  # 4*4


def convert_T2pose(T):
    # T: 4*4
    return [T[:3, :3], T[:3, 3]]


def convert_pose2angleAxis(pose):
    # pose: [R: 3*3, t: 3]
    angle_axis_torch = transforms.so3_log_map(
        torch.from_numpy(np.copy(pose[0])).unsqueeze(0)
    )  # 1*3
    angle_axis = angle_axis_torch.squeeze().numpy()  # 3
    return np.concatenate([angle_axis[None], pose[1][None]], axis=1)  # 1*6


def convert_T2angleAxis(T):
    # T: 4*4
    pose = convert_T2pose(T)
    return convert_pose2angleAxis(pose)


def get_pose_from_colmap_image(image):
    # return: [R: numpy.array 3*3, t: numpy.array 3]
    qvec = image.qvec
    R = qvec2rotmat(qvec)  # 3*3
    t = image.tvec  # 3
    return [R, t]


def get_intrinsic_from_colmap_camera(camera):
    model = camera.model
    if model == 'SIMPLE_RADIAL':
        focal = camera.params[0]
        x0 = camera.params[1]
        y0 = camera.params[2]
        intrinsic = np.array([[focal, 0, x0], [0, focal, y0], [0, 0, 1]])
    elif model == 'PINHOLE':
        focal0 = camera.params[0]
        focal1 = camera.params[1]
        x0 = camera.params[2]
        y0 = camera.params[3]
        intrinsic = np.array([[focal0, 0, x0], [0, focal1, y0], [0, 0, 1]])
    else:
        raise NotImplementedError

    return intrinsic


def project_point_cloud_to_image(intrinsic, pose, point_cloud):
    """
    Parameters:
    -------------
    intrinisc: np.array 3*3
    pose: [R: np.array 3*3, t: np.array 3]
    point_cloud: np.array N*3

    Return:
    ----------
    keypoints: N*2
    depth: N
    """
    R, t = pose
    point_cloud_f = (
        R @ point_cloud.T + t[:, None]
    )  # 3*N point cloud in camera coordiante
    point_cloud_rpj = (intrinsic @ point_cloud_f).T  # N*3

    keypoints = point_cloud_rpj[:, :2] / (point_cloud_rpj[:, [2]] + 1e-4)
    depth = point_cloud_rpj[:, 2]

    return keypoints, depth
