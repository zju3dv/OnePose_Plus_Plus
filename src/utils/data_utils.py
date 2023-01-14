import cv2
import torch
import numpy as np
from pathlib import Path


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans



def pad_features(features, num_leaf):
    num_features = features.shape[0]
    feature_dim = features.shape[1]
    n_pad = num_leaf - num_features

    if n_pad <= 0:
        features = features[:num_leaf]
    else:
        features = torch.cat(
            [features, torch.ones((num_leaf - num_features, feature_dim))], dim=0
        )

    return features.T


def pad_scores(scores, num_leaf):
    num_scores = scores.shape[0]
    n_pad = num_leaf - num_scores

    if n_pad <= 0:
        scores = scores[:num_leaf]
    else:
        scores = torch.cat([scores, torch.zeros((num_leaf - num_scores, 1))], dim=0)

    return scores


def avg_features(features):
    ret_features = torch.mean(features, dim=0).reshape(-1, 1)
    return ret_features


def avg_scores(scores):
    ret_scores = torch.mean(scores, dim=0).reshape(-1, 1)
    return ret_scores


def pad_keypoints3d_top_n(keypoints, n_target_kpts):
    """ Pad or truncate orig 3d keypoints to fixed size."""
    n_pad = n_target_kpts - keypoints.shape[0]

    if n_pad < 0:
        keypoints = keypoints[:n_target_kpts]  # [n_target_kpts: 3]

    else:
        while n_pad > 0:
            rand_kpts_x = torch.rand(n_pad, 1) - 0.5  # zero mean
            rand_kpts_y = torch.rand(n_pad, 1) - 0.5  # zero mean
            rand_kpts_z = torch.rand(n_pad, 1) - 0.5  # zero mean
            rand_kpts = torch.cat(
                [rand_kpts_x, rand_kpts_y, rand_kpts_z], dim=1
            )  # [n_pad, 3]

            exist = (rand_kpts[:, None, :] == keypoints[None, :, :]).all(-1).any(1)
            kept_kpts = rand_kpts[~exist]  # [n_kept, 3]
            n_pad -= len(kept_kpts)

            if len(kept_kpts) > 0:
                keypoints = torch.cat([keypoints, kept_kpts], dim=0)

    return keypoints

def pad_keypoints3d_according_to_assignmatrix(keypoints, n_target_kpts, assignmatrix):
    """ Pad or truncate orig 3d keypoints to fixed size."""
    # assignmatrix: [2, N]
    n_pad = n_target_kpts - keypoints.shape[0]

    if n_pad < 0:
        assignmatrix = assignmatrix.long()
        # Aim to keep all matches in assignmatrix
        # Get matches 3D points coord:
        keypoints_idx = torch.arange(keypoints.shape[0])
        unmatched_mask = torch.ones((keypoints.shape[0],), dtype=torch.bool)  # All True
        matched_3D_points = keypoints[assignmatrix[1]]  # N*3
        unmatched_mask[assignmatrix[1]] = False
        unmatched_3D_points = keypoints[unmatched_mask]

        num_of_pad = n_target_kpts - matched_3D_points.shape[0]
        assert num_of_pad >= 0

        unmatches_3D_random_index = torch.randint(
            unmatched_3D_points.shape[0], (num_of_pad,)
        )
        keypoints_padded = torch.cat(
            [matched_3D_points, unmatched_3D_points[unmatches_3D_random_index]]
        )  # [n_targets_kpts, 3]

        # Shuffle: (Maybe not useful?)
        shuffle_index = torch.randperm(keypoints_padded.shape[0])
        inverse_shuffle_index = torch.zeros((shuffle_index.shape[0]), dtype=torch.long)
        inverse_shuffle_index[shuffle_index] = torch.arange(shuffle_index.shape[0])
        keypoints_padded_shuffled = keypoints_padded[shuffle_index]

        # Build index for later selection:
        index = torch.cat(
            [assignmatrix[1], keypoints_idx[unmatched_mask][unmatches_3D_random_index]]
        )
        index_shuffled = index[shuffle_index]

        # Update assignmatrix
        assignmatrix[1, :] = inverse_shuffle_index[: matched_3D_points.shape[0]]
        keypoints = keypoints_padded_shuffled

    else:
        while n_pad > 0:
            rand_kpts_x = torch.rand(n_pad, 1) - 0.5  # zero mean
            rand_kpts_y = torch.rand(n_pad, 1) - 0.5  # zero mean
            rand_kpts_z = torch.rand(n_pad, 1) - 0.5  # zero mean
            rand_kpts = torch.cat(
                [rand_kpts_x, rand_kpts_y, rand_kpts_z], dim=1
            )  # [n_pad, 3]

            exist = (rand_kpts[:, None, :] == keypoints[None, :, :]).all(-1).any(1)
            kept_kpts = rand_kpts[~exist]  # [n_kept, 3]
            n_pad -= len(kept_kpts)

            if len(kept_kpts) > 0:
                keypoints = torch.cat([keypoints, kept_kpts], dim=0)

        index_shuffled = None

    return keypoints, assignmatrix, index_shuffled


def pad_features3d_top_n(descriptors, scores, n_target_shape):
    """ Pad or truncate orig 3d feature(descriptors and scores) to fixed size."""
    dim = descriptors.shape[0]
    n_pad = n_target_shape - descriptors.shape[1]

    if n_pad < 0:
        descriptors = descriptors[:, :n_target_shape]
        scores = scores[:n_target_shape, :]
    else:
        descriptors = torch.cat([descriptors, torch.ones(dim, n_pad)], dim=-1)
        scores = torch.cat([scores, torch.zeros(n_pad, 1)], dim=0)

    return descriptors, scores


def pad_features3d_according_to_assignmatrix(
    descriptors, scores, n_target_shape, padding_index
):
    """ Pad or truncate orig 3d feature(descriptors and scores) to fixed size."""
    dim = descriptors.shape[0]
    n_pad = n_target_shape - descriptors.shape[1]

    if n_pad < 0:
        descriptors = descriptors[:, padding_index]
        scores = scores[padding_index, :]
    else:
        descriptors = torch.cat([descriptors, torch.ones((dim, n_pad))], dim=-1)
        scores = torch.cat([scores, torch.zeros((n_pad, 1))], dim=0)

    return descriptors, scores


def pad_keypoints3d_random(keypoints, n_target_kpts):
    """ Pad or truncate orig 3d keypoints to fixed size."""
    # NOTE: For val
    n_pad = n_target_kpts - keypoints.shape[0]

    if n_pad < 0:
        selected_index = torch.randint(keypoints.shape[0], (n_target_kpts,))
        keypoints = keypoints[selected_index]
    else:
        selected_index = torch.arange(keypoints.shape[0])

    return keypoints, selected_index

def pad_features3d_random(
    descriptors, scores, n_target_shape, padding_index
):
    """ Pad or truncate orig 3d feature(descriptors and scores) to fixed size."""
    # NOTE: For val
    dim = descriptors.shape[0]
    n_pad = n_target_shape - descriptors.shape[1]

    if n_pad < 0:
        descriptors = descriptors[:, padding_index]
        scores = scores[padding_index, :]

    return descriptors, scores

def get_image_crop_resize(image, box, resize_shape):
    """Crop image according to the box, and resize the cropped image to resize_shape
    @param image: the image waiting to be cropped
    @param box: [x0, y0, x1, y1]
    @param resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    scale = np.array([box[2] - box[0], box[3] - box[1]])

    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    image_crop = cv2.warpAffine(
        image, trans_crop, (resize_w, resize_h), flags=cv2.INTER_LINEAR
    )

    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)
    return image_crop, trans_crop_homo


def get_K_crop_resize(box, K_orig, resize_shape):
    """Update K (crop an image according to the box, and resize the cropped image to resize_shape) 
    @param box: [x0, y0, x1, y1]
    @param K_orig: [3, 3] or [3, 4]
    @resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    scale = np.array([box[2] - box[0], box[3] - box[1]])  # w, h

    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)

    if K_orig.shape == (3, 3):
        K_orig_homo = np.concatenate([K_orig, np.zeros((3, 1))], axis=-1)
    else:
        K_orig_homo = K_orig.copy()
    assert K_orig_homo.shape == (3, 4)

    K_crop_homo = trans_crop_homo @ K_orig_homo  # [3, 4]
    K_crop = K_crop_homo[:3, :3]

    return K_crop, K_crop_homo

def get_K(intrin_file):
    assert Path(intrin_file).exists()
    with open(intrin_file, 'r') as f:
        lines = f.readlines()
    intrin_data = [line.rstrip('\n').split(':')[1] for line in lines]
    fx, fy, cx, cy = list(map(float, intrin_data))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    return K, K_homo
