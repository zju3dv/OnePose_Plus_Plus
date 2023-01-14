from wis3d import Wis3D as Vis3D
import open3d as o3d
import os.path as osp
import os
from loguru import logger
import json
import numpy as np

# Use vis3d to visualize camera and point cloud
def save_point_cloud(point_cloud, save_path):
    """
    Parameters:
    ------------
    point_cloud : np.array
        [N,3]
    save_path : str
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(save_path, pcd)


def vis_cameras_point_clouds(
    pointcloud,
    dump_dir,
    name,
    pointcloud_color=None,
    old_pointcloud_coord=None,
    old_pointcloud_color=None,
):
    # pointclouds: N*3
    # cameras: N*4*4
    vis3d = Vis3D(dump_dir, name, xyz_pattern=("x", "-y", "-z"))
    os.makedirs(dump_dir, exist_ok=True)

    vis3d.add_point_cloud(
        np.array(pointcloud),
        np.array(pointcloud_color),
        name="point_clouds after refinement",
    )

    if old_pointcloud_coord is not None:
        # old_pointcloud_coord[:, [0, 2]] *= -1  # Consistent with refined pointcloud
        vis3d.add_point_cloud(
            np.array(old_pointcloud_coord),
            np.array(old_pointcloud_color)
            if old_pointcloud_color is not None
            else None,
            name="point_clouds_before_refinement",
        )

    logger.info(f"vis3d saved to {dump_dir}")
