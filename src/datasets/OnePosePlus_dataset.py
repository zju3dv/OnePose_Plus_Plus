from loguru import logger

try:
    import ujson as json
except ImportError:
    import json
import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from kornia import homography_warp, normalize_homography, normal_transform_pixel
from src.utils.data_io import read_grayscale
from src.utils import data_utils
from src.utils.sample_homo import sample_homography_sap


class OnePosePlusDataset(Dataset):
    def __init__(
        self,
        anno_file,
        pad=True,
        img_pad=False,
        img_resize=512,
        coarse_scale=1 / 8,
        df=8,
        shape3d=10000,
        percent=1.0,
        split="train",
        load_pose_gt=False,
        load_3d_coarse_feature=False,
        image_warp_adapt=False,
        augmentor=None
    ):
        super(Dataset, self).__init__()

        self.split = split
        self.coco = COCO(anno_file)
        self.anns = np.array(self.coco.getImgIds())

        logger.info(f"Use {percent * 100}% data")
        sample_inverval = int(1 / percent)
        self.anns = self.anns[::sample_inverval]

        self.load_pose_gt = load_pose_gt
        self.image_warp_adapt = image_warp_adapt

        # 3D point cloud part
        self.pad = pad
        self.load_3d_coarse = load_3d_coarse_feature
        self.shape3d = shape3d

        # 2D query image part
        self.img_pad = img_pad
        self.img_resize = img_resize
        self.df = df
        self.coarse_scale = coarse_scale

        self.augmentor = augmentor

    def read_anno2d(self, anno2d_file):
        """ Read (and pad) 2d info"""
        with open(anno2d_file, "r") as f:
            data = json.load(f)

        keypoints2d = torch.Tensor(data["keypoints2d"])  # [n, 2]
        scores2d = torch.Tensor(data["scores2d"])  # [n, 1]
        assign_matrix = torch.Tensor(data["assign_matrix"])  # [2, k]

        num_2d_orig = keypoints2d.shape[0]

        return keypoints2d, scores2d, assign_matrix, num_2d_orig

    def read_anno3d(
        self,
        avg_anno3d_file,
        pad=True,
        assignmatrix=None,
        load_3d_coarse=False,
    ):
        """ Read(and pad) 3d info"""
        avg_data = np.load(avg_anno3d_file)

        keypoints3d = torch.Tensor(avg_data["keypoints3d"])  # [m, 3]
        avg_descriptors3d = torch.Tensor(avg_data["descriptors3d"])  # [dim, m]
        avg_scores = torch.Tensor(avg_data["scores3d"])  # [m, 1]

        if load_3d_coarse:
            avg_anno3d_coarse_file = (
                osp.splitext(avg_anno3d_file)[0]
                + "_coarse"
                + osp.splitext(avg_anno3d_file)[1]
            )
            avg_coarse_data = np.load(avg_anno3d_coarse_file)
            avg_coarse_descriptors3d = torch.Tensor(
                avg_coarse_data["descriptors3d"]
            )  # [dim, m]
            avg_coarse_scores = torch.Tensor(avg_coarse_data["scores3d"])  # [m, 1]

        else:
            avg_coarse_descriptors3d = None

        if pad:
            if self.split == "train":
                if assignmatrix is not None:
                    (
                        keypoints3d,
                        assignmatrix,
                        padding_index,
                    ) = data_utils.pad_keypoints3d_according_to_assignmatrix(
                        keypoints3d, self.shape3d, assignmatrix=assignmatrix
                    )
                    (
                        avg_descriptors3d,
                        avg_scores,
                    ) = data_utils.pad_features3d_according_to_assignmatrix(
                        avg_descriptors3d, avg_scores, self.shape3d, padding_index
                    )

                    if avg_coarse_descriptors3d is not None:
                        (
                            avg_coarse_descriptors3d,
                            avg_coarse_scores,
                        ) = data_utils.pad_features3d_according_to_assignmatrix(
                            avg_coarse_descriptors3d,
                            avg_coarse_scores,
                            self.shape3d,
                            padding_index,
                        )
                else:
                    keypoints3d = data_utils.pad_keypoints3d_top_n(
                        keypoints3d, self.shape3d
                    )
                    avg_descriptors3d, avg_scores = data_utils.pad_features3d_top_n(
                        avg_descriptors3d, avg_scores, self.shape3d
                    )

                    if avg_coarse_descriptors3d is not None:
                        (
                            avg_coarse_descriptors3d,
                            avg_coarse_scores,
                        ) = data_utils.pad_features3d_top_n(
                            avg_coarse_descriptors3d, avg_coarse_scores, self.shape3d
                        )
            else:
                (keypoints3d, padding_index,) = data_utils.pad_keypoints3d_random(
                    keypoints3d, self.shape3d
                )
                (avg_descriptors3d, avg_scores,) = data_utils.pad_features3d_random(
                    avg_descriptors3d, avg_scores, self.shape3d, padding_index
                )

                if avg_coarse_descriptors3d is not None:
                    (
                        avg_coarse_descriptors3d,
                        avg_coarse_scores,
                    ) = data_utils.pad_features3d_random(
                        avg_coarse_descriptors3d,
                        avg_coarse_scores,
                        self.shape3d,
                        padding_index,
                    )

        return (
            keypoints3d,
            avg_descriptors3d,
            avg_coarse_descriptors3d,
            avg_scores,
            assignmatrix,  # Update assignmatrix
        )

    def build_assignmatrix(
        self, keypoints2D_coarse, keypoints2D_fine, assign_matrix, pad=True
    ):
        """
        Build assign matrix for coarse and fine
        Coarse assign matrix: store 0 or 1
        Fine matrix: store corresponding 2D fine location in query image of the matched coarse grid point (N*M*2)
        Reshape assign matrix (from 2xk to nxm)
        """
        assign_matrix = assign_matrix.long()

        if pad:
            conf_matrix = torch.zeros(
                self.shape3d, self.n_query_coarse_grid, dtype=torch.int16
            )  # [n_pointcloud, n_coarse_grid]

            fine_location_matrix = torch.full(
                (self.shape3d, self.n_query_coarse_grid, 2), -50, dtype=torch.float
            )

            # Padding
            valid = assign_matrix[1] < self.shape3d
            assign_matrix = assign_matrix[:, valid]

            # Get grid coordinate for query image
            keypoints_idx = assign_matrix[0]
            keypoints2D_coarse_selected = keypoints2D_coarse[keypoints_idx]

            keypoints2D_fine_selected = keypoints2D_fine[keypoints_idx]

            # Get j_id of coarse keypoints in grid
            keypoints2D_coarse_selected_rescaled = (
                keypoints2D_coarse_selected
                / self.query_img_scale[[1, 0]]
                * self.coarse_scale
            )
            keypoints2D_coarse_selected_rescaled = (
                keypoints2D_coarse_selected_rescaled.round()
            )
            unique, counts = np.unique(
                keypoints2D_coarse_selected_rescaled, return_counts=True, axis=0
            )
            if unique.shape[0] != keypoints2D_coarse_selected_rescaled.shape[0]:
                logger.warning("Keypoints duplicate! Problem exists")

            j_ids = (
                keypoints2D_coarse_selected_rescaled[:, 1] * self.w_c  # y
                + keypoints2D_coarse_selected_rescaled[:, 0]  # x
            )
            j_ids = j_ids.long()

            invalid_mask = j_ids > conf_matrix.shape[1]
            j_ids = j_ids[~invalid_mask]
            assign_matrix = assign_matrix[:, ~invalid_mask]
            keypoints2D_fine_selected = keypoints2D_fine_selected[~invalid_mask]

            conf_matrix[assign_matrix[1], j_ids] = 1
            fine_location_matrix[assign_matrix[1], j_ids] = keypoints2D_fine_selected

        else:
            raise NotImplementedError

        return conf_matrix, fine_location_matrix

    def get_intrin_by_color_pth(self, img_path):
        img_ext = osp.splitext(img_path)[1]
        intrin_path = img_path.replace("/color/", "/intrin_ba/").replace(img_ext, ".txt")
        K_crop = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]
        return K_crop

    def get_gt_pose_by_color_pth(self, img_path):
        img_ext = osp.splitext(img_path)[1]
        gt_pose_path = img_path.replace("/color/", "/poses_ba/").replace(img_ext, ".txt")
        pose_gt = torch.from_numpy(np.loadtxt(gt_pose_path))  # [4*4]
        return pose_gt

    def read_anno(self, img_id, image_warp_adapt=False):
        """
        read image, 2d info and 3d info.
        pad 2d info and 3d info to a constant size.
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        color_path = self.coco.loadImgs(int(img_id))[0]["img_file"]

        query_img, query_img_scale, query_img_mask = read_grayscale(
            color_path,
            resize=self.img_resize,
            pad_to=self.img_resize if self.img_pad else None,
            ret_scales=True,
            ret_pad_mask=True,
            df=self.df,
            augmentor=self.augmentor
        )

        self.h_origin = query_img.shape[1] * query_img_scale[0]
        self.w_origin = query_img.shape[2] * query_img_scale[1]
        self.query_img_scale = query_img_scale
        self.h_i = query_img.shape[1]
        self.w_i = query_img.shape[2]
        self.h_c = int(self.h_i * self.coarse_scale)
        self.w_c = int(self.w_i * self.coarse_scale)

        self.n_query_coarse_grid = int(self.h_c * self.w_c)

        data = {}

        if query_img_mask is not None:
            data.update({"query_image_mask": query_img_mask})  # [h*w]

        if self.load_pose_gt:
            K_crop = self.get_intrin_by_color_pth(color_path)
            pose_gt = self.get_gt_pose_by_color_pth(color_path)

            data.update({"query_intrinsic": K_crop, "query_pose_gt": pose_gt})

        if self.split == "train":
            # For query image GT correspondences
            anno2d_file = anno["anno2d_file"]

            anno2d_coarse_file = anno2d_file.replace(
                "/anno_loftr/", "/anno_loftr_coarse/"
            )
            (
                keypoints2d_coarse,
                scores2d,
                assign_matrix,
                num_2d_orig,
            ) = self.read_anno2d(anno2d_coarse_file)

        else:
            assign_matrix = None

        avg_anno3d_file = anno["avg_anno3d_file"]

        (
            keypoints3d,
            avg_descriptors3d,
            avg_coarse_descriptors3d,
            avg_scores3d,
            assign_matrix,
        ) = self.read_anno3d(
            avg_anno3d_file,
            pad=self.pad,
            assignmatrix=assign_matrix,
            load_3d_coarse=self.load_3d_coarse,
        )

        data.update(
            {
                "keypoints3d": keypoints3d,  # [n2, 3]
                "descriptors3d_db": avg_descriptors3d,  # [dim, n2]
                "scores3d_db": avg_scores3d.squeeze(1),  # [n2]
                "query_image": query_img,  # [1*h*w]
                "query_image_scale": query_img_scale,  # [2]
                "query_image_path": color_path
            }
        )

        if avg_coarse_descriptors3d is not None:
            data.update(
                {
                    "descriptors3d_coarse_db": avg_coarse_descriptors3d,  # [dim, n2]
                }
            )

        if self.split == "train":
            assign_matrix = assign_matrix.long()
            mkpts_3d = keypoints3d[assign_matrix[1, :]]  # N*3
            R = pose_gt[:3, :3].to(torch.float)  # 3*3
            t = pose_gt[:3, [3]].to(torch.float)  # 3*1
            K_crop = K_crop.to(torch.float)
            keypoints2d_fine = torch.zeros(
                (keypoints2d_coarse.shape[0], 2), dtype=torch.float
            )

            # Project 3D pointcloud to make fine GT
            mkpts_3d_camera = R @ mkpts_3d.transpose(1, 0) + t
            mkpts_proj = (K_crop @ mkpts_3d_camera).transpose(1, 0)  # N*3
            mkpts_proj = mkpts_proj[:, :2] / (mkpts_proj[:, [2]] + 1e-6)

            # Sample random homography and transform image to get more source images
            if image_warp_adapt:
                homo_sampled = sample_homography_sap(self.h_i, self.w_i)
                homo_sampled_normed = normalize_homography(
                    torch.from_numpy(homo_sampled[None]).to(torch.float32),
                    (self.h_i, self.w_i),
                    (self.h_i, self.w_i),
                )
                homo_warpped_image = homography_warp(
                    data["query_image"][None],
                    torch.linalg.inv(homo_sampled_normed),
                    (self.h_i, self.w_i),
                )[
                    0
                ]  # 1*h*w

                # Warp kpts:
                norm_pixel_mat = normal_transform_pixel(self.h_i, self.w_i)
                mkpts_proj_normed = (
                    norm_pixel_mat[0].numpy()
                    @ (
                        np.concatenate(
                            [mkpts_proj, np.ones((mkpts_proj.shape[0], 1))], axis=-1
                        )
                    ).T
                ).astype(np.float32)
                mkpts_proj_normed_warpped = (
                    norm_pixel_mat[0].inverse()
                    @ homo_sampled_normed[0]
                    @ mkpts_proj_normed
                ).T  # N*3

                mkpts_proj_normed_warpped[:, :2] /= mkpts_proj_normed_warpped[
                    :, [2]
                ]  # NOTE: Important! [:, 2] is not all 1!
                mkpts_proj = mkpts_proj_normed_warpped[:, :2]  # N*2

                out_of_boundry_mask = (
                    (mkpts_proj[:, 0] < 0)
                    | (mkpts_proj[:, 0] > (self.w_i - 1))
                    | (mkpts_proj[:, 1] < 0)
                    | (mkpts_proj[:, 1] > (self.h_i - 1))
                )
                mkpts_proj = mkpts_proj[~out_of_boundry_mask]
                assign_matrix = assign_matrix[:, ~out_of_boundry_mask]
                K_crop_warpped = (
                    torch.from_numpy(homo_sampled).to(torch.float32) @ K_crop
                )  # FIXME: incorrect! Shouldn't H \times K directly
                data.update(
                    {
                        "query_image": homo_warpped_image,
                        "query_intrinsic": K_crop_warpped,
                    }
                )

            mkpts_proj_rounded = (
                mkpts_proj / int(1 / self.coarse_scale)
            ).round() * int(
                1 / self.coarse_scale
            )
            invalid = (
                (mkpts_proj_rounded[:, 0] < 0)
                | (mkpts_proj_rounded[:, 0] > query_img.shape[-1] - 1)
                | (mkpts_proj_rounded[:, 1] < 0)
                | (mkpts_proj_rounded[:, 1] > query_img.shape[-2] - 1)
            )
            mkpts_proj_rounded = mkpts_proj_rounded[~invalid]
            mkpts_proj = mkpts_proj[~invalid]
            assign_matrix = assign_matrix[:, ~invalid]

            unique, index = np.unique(mkpts_proj_rounded, return_index=True, axis=0)

            mkpts_proj_rounded = mkpts_proj_rounded[index]
            mkpts_proj = mkpts_proj[index]
            assign_matrix = assign_matrix[:, index]
            keypoints2d_coarse[assign_matrix[0, :]] = mkpts_proj_rounded

            keypoints2d_fine[assign_matrix[0, :]] = mkpts_proj

            (conf_matrix, fine_location_matrix) = self.build_assignmatrix(
                keypoints2d_coarse, keypoints2d_fine, assign_matrix, pad=self.pad
            )

            data.update(
                {
                    "conf_matrix_gt": conf_matrix,  # [n_point_cloud, n_query_coarse_grid] Used for coarse GT
                    "fine_location_matrix_gt": fine_location_matrix,  # [n_point_cloud, n_query_coarse_grid, 2] (x,y)
                }
            )

        return data

    def __getitem__(self, index):
        if not self.image_warp_adapt:
            img_id = self.anns[index]
            data = self.read_anno(img_id)
        else:
            img_id = self.anns[index // 2]
            data = self.read_anno(img_id, image_warp_adapt=(index % 2) != 0)
        return data

    def __len__(self):
        return len(self.anns) if not self.image_warp_adapt else len(self.anns) * 2
