from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from ..utils.geometry_utils import *


class ConstructOptimizationData(Dataset):
    def __init__(self, colmap_image_dataset, fine_match_results_dict) -> None:
        super().__init__()

        self.colmap_image_dataset = colmap_image_dataset
        self.colmap_frame_dict = colmap_image_dataset.colmap_frame_dict
        self.colmap_3ds = colmap_image_dataset.colmap_3ds
        self.colmap_images = colmap_image_dataset.colmap_images
        self.colmap_cameras = colmap_image_dataset.colmap_cameras
        self.point_cloud_assigned_imgID_kptsID_list = list(
            colmap_image_dataset.point_cloud_assigned_imgID_kptID.items()
        )

        self.fine_match_results_dict = fine_match_results_dict

        self.max_track_length = 0
        for point3D in self.colmap_3ds.values():
            if point3D.image_ids.shape[0] > self.max_track_length:
                self.max_track_length = point3D.image_ids.shape[0]
        
        self.padding_data = True # padding to max track length for fast dataloading.
        

    def __len__(self):
        return len(self.point_cloud_assigned_imgID_kptsID_list)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_single_item(index)

    def _get_single_item(self, index):
        point_cloudID, assigned_state = self.point_cloud_assigned_imgID_kptsID_list[
            index
        ]
        assigned_colmap_frameID, assigned_keypoint_index = assigned_state

        image_ids = self.colmap_3ds[point_cloudID].image_ids.tolist()
        point2D_idxs = self.colmap_3ds[point_cloudID].point2D_idxs.tolist()

        pairs_dict = {}  # {"query_id-ref_id": keypoint_idx_in_ref_frame}
        for image_id, kpt_id in zip(image_ids, point2D_idxs):
            if image_id != assigned_colmap_frameID:
                pairs_dict[
                    "-".join([str(assigned_colmap_frameID), str(image_id)])
                ] = kpt_id
        query_kpt_idx = assigned_keypoint_index

        initial_depth = self.colmap_frame_dict[assigned_colmap_frameID][
            "initial_depth"
        ][
            np.array([assigned_keypoint_index])
        ]  # 1
        intrinsic0 = self.colmap_frame_dict[assigned_colmap_frameID]["intrinsic"][
            None
        ]  # 1*3*3 intrinsics from colmap
        intrinsic1 = []  # N*3*3

        left_colmap_ids = []  # N
        right_colmap_ids = []  # N
        mkpts0_c = []  # N*2
        mkpts1_c = []  # N*2
        mkpts1_f = []  # N*2
        scale0 = []  # N*2
        scale1 = []  # N*2

        for pair_name, ref_kpt_idx in pairs_dict.items():
            assert pair_name in self.fine_match_results_dict
            left_colmap_id, right_colmap_id = pair_name.split("-")
            fine_match_results = self.fine_match_results_dict[pair_name]
            index = np.argwhere(fine_match_results["mkpts0_idx"] == query_kpt_idx)
            assert len(index) == 1, len(index)
            index = np.squeeze(index)
            mkpts0_c.append(fine_match_results["mkpts0_c"][index])
            mkpts1_c.append(fine_match_results["mkpts1_c"][index])
            mkpts1_f.append(fine_match_results["mkpts1_f"][index])
            intrinsic1.append(self.colmap_frame_dict[int(right_colmap_id)]["intrinsic"])
            left_colmap_ids.append(
                np.array([int(left_colmap_id)])
            )  # left frame colmap id, used to index pose
            right_colmap_ids.append(
                np.array([int(right_colmap_id)])
            )  # right frame colmap id, used to index pose

            # Get scale
            scale0.append(np.squeeze(fine_match_results["scale0"], axis=0))
            scale1.append(np.squeeze(fine_match_results["scale1"], axis=0))

        (
            intrinsic1,
            mkpts0_c,
            mkpts1_c,
            mkpts1_f,
            scale0,
            scale1,
            left_colmap_ids,
            right_colmap_ids,
        ) = map(
            lambda a: np.stack(a),
            [
                intrinsic1,
                mkpts0_c,
                mkpts1_c,
                mkpts1_f,
                scale0,
                scale1,
                left_colmap_ids,
                right_colmap_ids,
            ],
        )
        num_query = mkpts1_c.shape[0]
        point_cloud_id = np.full_like(initial_depth, point_cloudID)

        data = {
            "intrinsic0": torch.from_numpy(np.copy(intrinsic0)).expand(
                num_query, -1, -1
            ),  # from [1*3*3] to [N*3*3]
            "intrinsic1": torch.from_numpy(intrinsic1),  # [N*3*3]
            "mkpts0_c": torch.from_numpy(mkpts0_c),  # [N*2]
            "mkpts1_c": torch.from_numpy(mkpts1_c),  # [N*2]
            "mkpts1_f": torch.from_numpy(mkpts1_f),  # [N*2]
            "scale0": torch.from_numpy(scale0),  # [N*2]
            "scale1": torch.from_numpy(scale1),  # [N*2]
            "left_colmap_ids": torch.from_numpy(left_colmap_ids).squeeze(-1),  # np.array [N]
            "right_colmap_ids": torch.from_numpy(right_colmap_ids).squeeze(-1),  # np.array [N]
        }

        if self.padding_data:
            # Padding for multiprocess data loading
            if num_query < self.max_track_length:
                for key, value in data.items():
                    shape = list(value.shape)
                    shape[0] = self.max_track_length - num_query
                    value_padded = torch.cat(
                        [value, torch.zeros(shape)], dim=0
                    )  # max_track_length * ...
                    data[key] = value_padded

        data.update(
            {
                "depth": torch.from_numpy(initial_depth).unsqueeze(-1),  # [1*1]
                "point_cloud_id": torch.from_numpy(
                    point_cloud_id
                ),  # [1] related point cloud id
                "n_query": torch.tensor(
                    [num_query]
                ),  # [1] used to split data from padding
            }
        )
        return data 

