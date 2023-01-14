from torch.utils.data.dataset import Dataset
import numpy as np
import torch

class MatchingPairData(Dataset):
    """
    Construct image pair for refinement matching
    """

    def __init__(self, colmap_image_dataset) -> None:
        super().__init__()

        # Colmap info
        self.colmap_image_dataset = colmap_image_dataset
        self.colmap_frame_dict = colmap_image_dataset.colmap_frame_dict
        self.colmap_3ds = colmap_image_dataset.colmap_3ds
        self.colmap_images = colmap_image_dataset.colmap_images
        self.colmap_cameras = colmap_image_dataset.colmap_cameras

        self.all_pairs = []
        for colmap_frameID, colmap_frame_info in self.colmap_frame_dict.items():
            if colmap_frame_info["is_keyframe"]:
                for related_frameID in colmap_frame_info["related_frameID"]:
                    self.all_pairs.append([colmap_frameID, related_frameID])

    def buildDataPair(self, data0, data1):
        # data0: dict, data1: dict
        data = {}
        for i, data_part in enumerate([data0, data1]):
            for key, value in data_part.items():
                data[key + str(i)] = value
        assert (
            len(data) % 2 == 0
        ), "Build data pair error!"
        data["pair_names"] = (data["img_path0"], data["img_path1"])
        return data

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        left_img_id, right_img_id = self.all_pairs[index]  # colmap id

        # Get coarse matches
        left_kpts = []  # x,y
        right_kpts = []
        left_kpts_idx = []  # correspond to original index of left frame keypoints

        left_frame_info = self.colmap_frame_dict[left_img_id]
        valid_kpts_mask = left_frame_info["all_kpt_status"] >= 0
        valid_kpts_idxs = np.arange(left_frame_info["keypoints"].shape[0])[
            valid_kpts_mask
        ]
        valid_kpts = left_frame_info["keypoints"][valid_kpts_mask]
        related_3d_ids = left_frame_info["all_kpt_status"][valid_kpts_mask]
        for i, related_3d_id in enumerate(related_3d_ids.tolist()):
            related_index = np.argwhere(
                self.colmap_3ds[related_3d_id].image_ids == right_img_id
            )  # (1,1) or (0,1)
            if len(related_index) != 0:
                # successfully find!
                if len(related_index) != 1:
                    related_index = related_index[0]
                point2d_idx = self.colmap_3ds[related_3d_id].point2D_idxs[
                    np.squeeze(related_index).tolist()
                ]  # int
                left_kpts.append(
                    valid_kpts[i]
                )
                right_kpts.append(self.colmap_images[right_img_id].xys[point2d_idx])

                # Record left keypoints index in original frame keypoints
                (
                    self_img_id,
                    self_kpt_idx,
                ) = self.colmap_image_dataset.point_cloud_assigned_imgID_kptID[
                    related_3d_id
                ]
                assert self_img_id == left_img_id
                left_kpts_idx.append(valid_kpts_idxs[[i]])

        left_kpts = np.stack(left_kpts, axis=0)  # N*2
        right_kpts = np.stack(right_kpts, axis=0)  # N*2
        left_kpts_idx = np.concatenate(left_kpts_idx)  # N*1

        # Get images information
        left_id = self.colmap_image_dataset.colmapID2frameID_dict[
            left_img_id
        ]  # dataset image id
        right_id = self.colmap_image_dataset.colmapID2frameID_dict[right_img_id]
        left_image_dict = self.colmap_image_dataset[left_id]
        right_image_dict = self.colmap_image_dataset[right_id]

        pair_data = self.buildDataPair(left_image_dict, right_image_dict)

        pair_data.update(
            {
                "mkpts0_c": torch.from_numpy(left_kpts),
                "mkpts1_c": torch.from_numpy(right_kpts),
                "mkpts0_idx": torch.from_numpy(left_kpts_idx),
                "frame0_colmap_id": left_img_id,
                "frame1_colmap_id": right_img_id,
            }
        )

        return pair_data

