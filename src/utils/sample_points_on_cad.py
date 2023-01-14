import open3d as o3d
import os.path as osp
import numpy as np
from plyfile import PlyData

def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def sample_points_on_cad(cad_model, n_num=1000, save_p3d_path=None):
    """
    cad_model: str(path) or open3d mesh
    """
    if isinstance(cad_model, str):
        assert osp.exists(cad_model), f"CAD model mesh: {cad_model} not exists"
        mesh = o3d.io.read_triangle_mesh(cad_model)
    else:
        mesh = cad_model

    model_corners = get_model_corners(np.asarray(mesh.vertices))
    model_center = (np.max(model_corners, 0, keepdims=True) + np.min(model_corners, 0, keepdims=True)) / 2
    model_8corners_center = np.concatenate([model_corners, model_center], axis=0) # 9*3

    # Sample uniformly
    sampled_3D_points = mesh.sample_points_uniformly(n_num)

    # Save:
    if save_p3d_path is not None:
        o3d.io.write_point_cloud(save_p3d_path, sampled_3D_points)

    sampled_3D_points = np.asarray(sampled_3D_points.points)
    return sampled_3D_points.astype(np.float32), model_8corners_center.astype(np.float32) # 9*3

def load_points_from_cad(cad_model, max_num=-1, save_p3d_path=None):
    """
    cad_model: str(path) or open3d mesh
    """
    if isinstance(cad_model, str):
        assert osp.exists(cad_model), f"CAD model mesh: {cad_model} not exists"
        mesh = o3d.io.read_triangle_mesh(cad_model)
    else:
        mesh = cad_model

    model_corners = get_model_corners(np.asarray(mesh.vertices))
    model_center = (np.max(model_corners, 0, keepdims=True) + np.min(model_corners, 0, keepdims=True)) / 2
    model_8corners_center = np.concatenate([model_corners, model_center], axis=0) # 9*3

    # Sample uniformly
    # sampled_3D_points = o3d.geometry.sample_points_uniformly(mesh, n_num)
    vertices = np.asarray(mesh.vertices)
    if vertices.shape[0] > max_num and max_num != -1:
        sampled_3D_points = mesh.sample_points_uniformly(max_num)
        vertices = np.asarray(sampled_3D_points.points)

    # Save:
    if save_p3d_path is not None:
        o3d.io.write_point_cloud(save_p3d_path, vertices)

    return vertices.astype(np.float32), model_8corners_center.astype(np.float32) # 9*3

def model_diameter_from_bbox(bbox):
    """
    bbox: 8*3 or 9*3(including center at last row)
    """
    min_coord = bbox[0] # 3
    max_coord = bbox[7] # 3
    diameter = np.linalg.norm(max_coord - min_coord)
    return diameter

def get_all_points_on_model(cad_model_path):
    ply = PlyData.read(cad_model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    model = np.stack([x, y, z], axis=-1)
    return model