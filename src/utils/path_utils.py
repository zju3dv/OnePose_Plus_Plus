import os
import os.path as osp


"""
For each object, we store in the following directory format:

data_root:
    - box3d_corners.txt
    - seq1_root
        - intrinsics.txt
        - color/
        - poses_ba/
        - intrin_ba/
        - ......
    - seq2_root
    - ......
"""

def get_gt_pose_path_by_color(color_path, det_type='GT_box'):
    ext = osp.splitext(color_path)[1]
    if det_type == "GT_box":
        return color_path.replace("/color/", "/poses_ba/").replace(
            ext, ".txt"
        )
    elif det_type == 'feature_matching':
        return color_path.replace("/color_det/", "/poses_ba/").replace(
            ext, ".txt"
        )
    else:
        raise NotImplementedError

def get_img_full_path_by_color(color_path, det_type='GT_box'):
    if det_type == "GT_box":
        return color_path.replace("/color/", "/color_full/")
    elif det_type == 'feature_matching':
        return color_path.replace("/color_det/", "/color_full/")
    else:
        raise NotImplementedError

def get_intrin_path_by_color(color_path, det_type='GT_box'):
    if det_type == "GT_box":
        return color_path.replace("/color/", "/intrin_ba/").replace(
            ".png", ".txt"
        )
    elif det_type == 'feature_matching':
        return color_path.replace("/color_det/", "/intrin_det/").replace(
            ".png", ".txt"
        )
    else:
        raise NotImplementedError

def get_intrin_dir(seq_root):
    return osp.join(seq_root, "intrin_ba")

def get_gt_pose_dir(seq_root):
    return osp.join(seq_root, "poses_ba")

def get_intrin_full_path(seq_root):
    return osp.join(seq_root, "intrinsics.txt")

def get_3d_box_path(data_root):
    return osp.join(data_root, "box3d_corners.txt")

def get_test_seq_path(obj_root, last_n_seq_as_test=1):
    seq_names = os.listdir(obj_root)
    seq_names = [seq_name for seq_name in seq_names if '-' in seq_name]
    seq_ids = [int(seq_name.split('-')[-1]) for seq_name in seq_names if '-' in seq_name]
    
    test_obj_name = seq_names[0].split('-')[0]
    test_seq_ids = sorted(seq_ids)[(-1 * last_n_seq_as_test):]
    test_seq_paths = [osp.join(obj_root, test_obj_name + '-' + str(test_seq_id)) for test_seq_id in test_seq_ids]
    return test_seq_paths
