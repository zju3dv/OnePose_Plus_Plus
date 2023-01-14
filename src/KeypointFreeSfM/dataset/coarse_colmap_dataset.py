import os
import os.path as osp
from loguru import logger
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from src.utils.data_io import read_grayscale

from src.utils.colmap.read_write_model import (
    read_images_binary,
    read_cameras_binary,
    read_points3d_binary,
    qvec2rotmat,
    rotmat2qvec,
    write_model,
)
from ..post_optimization.utils.vis3d import vis_cameras_point_clouds
from ..post_optimization.utils.geometry_utils import *


class CoarseReconDataset(Dataset):
    def __init__(
        self,
        args,
        image_lists,
        covis_pairs,
        coarse_recon_results_dir,
        save_dir,
        pre_sfm=False,
        vis_path=None,
    ):
        """
        Parameters:
        ---------------
        image_lists: ['path/to/image/0.png', 'path/to/image/1.png, ...]
        covis_pairs: List or path
        colmap_results_dir: The directory contains images.bin(.txt) point3D.bin(.txt)...
        """
        super().__init__()
        self.img_list = image_lists

        self.coarse_recon_results_dir = coarse_recon_results_dir
        self.refined_save_dir = save_dir
        self.vis_path = vis_path

        self.img_resize = args["img_resize"]
        self.df = args["df"]
        self.feature_track_assignment_strategy = args[
            "feature_track_assignment_strategy"
        ]
        self.verbose = args["verbose"]
        self.state = True

        if isinstance(covis_pairs, list):
            self.pair_list = covis_pairs
        else:
            # Load pairs:
            with open(covis_pairs, "r") as f:
                self.pair_list = f.read().rstrip("\n").split("\n")

        self.frame_ids = list(range(len(self.img_list)))

        # Load colmap coarse results:
        is_coarse_valid = osp.exists(osp.join(coarse_recon_results_dir))
        assert (
            is_coarse_valid
        ), f"COLMAP is not valid, current COLMAP path: {osp.join(coarse_recon_results_dir)}"

        self.colmap_images = read_images_binary(
            osp.join(coarse_recon_results_dir, "images.bin")
        )
        self.colmap_3ds = read_points3d_binary(
            osp.join(coarse_recon_results_dir, "points3D.bin")
        )

        self.colmap_cameras = read_cameras_binary(
            osp.join(coarse_recon_results_dir, "cameras.bin")
        )

        # Verification:
        if (
            len(self.colmap_3ds) == 0
            or len(self.colmap_cameras) == 0
            or len(self.colmap_images) == 0
        ):
            self.state = False

        (
            self.frameId2colmapID_dict,
            self.colmapID2frameID_dict,
        ) = self.get_frameID2colmapID(
            self.frame_ids, self.img_list, self.colmap_images, pre_sfm=pre_sfm
        )

        # Get keyframes and feature track(3D points) assignment
        logger.info("Building keyframes begin....")
        if self.feature_track_assignment_strategy == "greedy":
            (
                self.keyframe_dict,
                self.point_cloud_assigned_imgID_kptID,
            ) = self.get_keyframes_greedy(
                self.colmap_images, self.colmap_3ds, verbose=self.verbose
            )
        else:
            raise NotImplementedError

        # Build depth and pose of each frame
        self.colmap_frame_dict = {}
        self.build_initial_depth_pose(self.colmap_frame_dict)

        # Extract corresponding frames and index of each keypoints
        self.extract_corresponding_frames(self.colmap_frame_dict)

    def build_initial_depth_pose(self, colmap_frame_dict):
        """
        Build initial pose for each registred frame, and build initial depth for each keyframe
        Update:
        initial_depth_pose_kpts: {colmap_frameID: {"initial_pose": [R: np.array 3*3, t: np.array 3],
                                              "intrinsic": np.array 3*3
                                              "keypoints": np.array [N_all,2] or None,
                                              "is_keyframe": bool, 
                                              "initial_depth": np.array N_all or None,
                                              "all_kpt_status": np.array N_all or None}}
        NOTE: None is when frame is not keyframe.
        NOTE: Intrinsic is from colmap
        """
        for colmap_frameID, colmap_image in self.colmap_images.items():
            initial_pose = get_pose_from_colmap_image(colmap_image)
            intrinisic = get_intrinsic_from_colmap_camera(
                self.colmap_cameras[colmap_image.camera_id]
            )
            keypoints = self.colmap_images[colmap_frameID].xys  # N_all*2

            if colmap_frameID in self.keyframe_dict:
                is_keyframe = True
                all_kpt_status = self.keyframe_dict[colmap_frameID]["state"]  # N_all

                occupied_mask = all_kpt_status >= 0
                point_cloud_idxs = all_kpt_status[
                    occupied_mask
                ]  # correspondence 3D index of keypoints, N
                point_cloud = np.concatenate(
                    [
                        self.colmap_3ds[point_cloud_idx].xyz[None]
                        for point_cloud_idx in point_cloud_idxs
                    ]
                )  # N*3
                reprojected_kpts, initial_depth_ = project_point_cloud_to_image(
                    intrinisic, initial_pose, point_cloud
                )

                initial_depth = np.ones((keypoints.shape[0],)) * -1
                initial_depth[occupied_mask] = initial_depth_

            else:
                initial_depth, keypoints, all_kpt_status = None, None, None
                is_keyframe = False

            colmap_frame_dict.update(
                {
                    colmap_frameID: {
                        "initial_pose": initial_pose,
                        "intrinsic": intrinisic,
                        "keypoints": keypoints,
                        "is_keyframe": is_keyframe,
                        "initial_depth": initial_depth,
                        "all_kpt_status": all_kpt_status,
                    }
                }
            )

    def extract_corresponding_frames(self, colmap_frame_dict):
        """
        Update: {related_frameID: list}
        """
        for colmap_frameID, frame_info in colmap_frame_dict.items():
            related_frameID = []
            if not frame_info["is_keyframe"]:
                continue
            all_kpt_status = frame_info["all_kpt_status"]
            point_cloud_idxs = all_kpt_status[all_kpt_status >= 0]
            for point_cloud_idx in point_cloud_idxs:
                # Get related feature track
                image_ids = self.colmap_3ds[point_cloud_idx].image_ids

                related_frameID.append(image_ids)

            all_related_frameID = np.concatenate(related_frameID)
            unique_frameID, counts = np.unique(all_related_frameID, return_counts=True)

            self_idx = np.squeeze(
                np.argwhere(unique_frameID == colmap_frameID)
            ).tolist()  # int
            unique_frameID = unique_frameID.tolist()
            unique_frameID.pop(self_idx)
            frame_info.update({"related_frameID": unique_frameID})

    def get_frameID2colmapID(
        self, frame_IDs, frame_names, colmap_images, pre_sfm=False
    ):
        # frame_id equal to frame_idx
        frameID2colmapID_dict = {}
        colmapID2frameID_dict = {}
        for frame_ID in frame_IDs:
            frame_name = frame_names[frame_ID]
            frame_name = osp.basename(frame_name) if pre_sfm else frame_name

            for colmap_image in colmap_images.values():
                if frame_name == colmap_image.name:
                    # Registrated scenario
                    frameID2colmapID_dict[frame_ID] = colmap_image.id
                    colmapID2frameID_dict[colmap_image.id] = frame_ID
                    break
            if frame_ID not in frameID2colmapID_dict:
                # -1 means not registrated
                frameID2colmapID_dict[frame_ID] = -1
        return frameID2colmapID_dict, colmapID2frameID_dict

    def get_keyframes_greedy(self, colmap_images, colmap_3ds, verbose=True):
        # Get keyframes by sorting num of keypoints and tracks of a frame.
        # Get each 3D point's correspondence image index and keypoint index

        # Build keypoints state and colmap state. -3 means robbed, -2 means unoccupied, -1 unregisted by colmap, -2 means robbed, >=0 means index of the 3D point(feature track)
        colmap_images_state = (
            {}
        )  # {colmap_imageID:{state: np.array [N], unoccupied_num: int n}}
        for id, colmap_image in colmap_images.items():
            colmap_images_state[id] = {}
            colmap_images_state[id]["state"] = -2 * np.ones(
                (colmap_image.xys.shape[0],)
            )  # [N], initial as all -2
            colmap_unregisted_mask = colmap_image.point3D_ids == -1
            colmap_images_state[id]["state"][
                colmap_unregisted_mask
            ] = -1  # set unregistred keypoints to -1
            colmap_images_state[id]["unoccupied_num"] = (
                (colmap_images_state[id]["state"] == -2)
            ).sum()
        colmap_3d_states = {}
        for point_cloudID, point_cloud in colmap_3ds.items():
            colmap_3d_states[point_cloudID] = (
                -1,
            )  # (-1,): unoccupied, (imageid, pointidx): occupied

        # Iterate to find keyframes:
        keyframe_dict = {}
        while not self._is_colmap_3d_empty(colmap_3d_states):
            assert len(colmap_images_state) != 0
            # Sort colmap images state:
            colmap_images_state = self._sort_colmap_images_state(colmap_images_state)

            # Set current frame with most keypoints to keyframe:
            current_keyframeID = list(colmap_images_state.keys())[0]
            current_selected_keyframe_state = colmap_images_state.pop(
                current_keyframeID
            )  # pop the first element of state dict
            # update current keyframe state
            occupied_keypoints_mask = current_selected_keyframe_state["state"] == -2
            current_selected_keyframe_state["state"][
                occupied_keypoints_mask
            ] = colmap_images[current_keyframeID].point3D_ids[occupied_keypoints_mask]
            keyframe_dict[current_keyframeID] = current_selected_keyframe_state

            # Update colmap_3d_state
            occupied_3d_ids = colmap_images[current_keyframeID].point3D_ids[
                occupied_keypoints_mask
            ]  # N'
            occupied_kpt_idx = np.arange(
                colmap_images[current_keyframeID].xys.shape[0]
            )[
                occupied_keypoints_mask
            ]  # N' Get index of keypoints in frame to update 3D point cloud state.
            for i, occupied_3d_id in enumerate(occupied_3d_ids):
                colmap_3d_states[occupied_3d_id] = (
                    current_keyframeID,
                    occupied_kpt_idx[i],
                )
                # Get feature track of this 3D point
                img_ids = colmap_3ds[occupied_3d_id].image_ids.tolist()
                point2d_idxs = colmap_3ds[occupied_3d_id].point2D_idxs.tolist()
                related_track = zip(img_ids, point2d_idxs)  # [[img_id, point2d_idx]]

                # Update other points' state in a track as robbed: -3
                for node in related_track:
                    img_id, point2d_idx = node
                    if img_id == current_keyframeID:
                        continue
                    original_point_state = colmap_images_state[img_id]["state"][
                        point2d_idx
                    ]
                    assert (
                        original_point_state != -1
                    ), "The state of the point in the track shouldn't be -1, bug exists!"
                    # update state
                    colmap_images_state[img_id]["state"][point2d_idx] = -3

            colmap_images_state = self._update_colmap_images_state_unoccupied_number(
                colmap_images_state
            )

        if verbose:
            logger.info(
                f"total {len(self.colmap_images)} frames registred, {len(keyframe_dict)} keyframes selected"
            )
            for id, item in keyframe_dict.items():
                print(
                    f"id:{id}, total: {sum(item['state'] != -1)} / {self.colmap_images[id].xys.shape[0]} 2D points registrated, possess {sum(item['state'] >= 0)} feature tracks, {sum(item['state'] == -3)} points robbed"
                )
        return keyframe_dict, colmap_3d_states

    def update_optimize_results_to_colmap(
        self, results_dict, visualize=False
    ):
        """
        Update optimized pose and depth to colmap format and save
        Parameters:
        --------------
        result_dict:{
            "pose": [R: np.array n_frames*3*3, t: np.array n_frames*3],
            "colmap_frame_ids": np.array n_frames,
            "depth": np.array n_point_clouds,
            "point_cloud_id": np.array n_point_clouds
        }
        """

        # Save old camera pose:
        old_pose = []
        for id, image in self.colmap_images.items():
            old_pose.append(convert_pose2T(get_pose_from_colmap_image(image)))

        # Update pose and keypoints:
        R_all, t_all = results_dict["pose"]  # n_frame*3*3, n_frames*3
        T_all = []
        for i, colmap_frame_id in enumerate(results_dict["colmap_frame_ids"].tolist()):
            R = R_all[i]  # [3*3]
            t = t_all[i]  # [3]
            qvec = rotmat2qvec(R)
            self.colmap_images[colmap_frame_id] = self.colmap_images[
                colmap_frame_id
            ]._replace(qvec=qvec, tvec=t)
            T_all.append(convert_pose2T([R, t]))

        # Save old point clouds
        old_point_cloud = []
        old_point_cloud_color = []
        for id, pointcloud in self.colmap_3ds.items():
            old_point_cloud.append(pointcloud.xyz)
            old_point_cloud_color.append(pointcloud.rgb)
        point_cloud = []
        point_cloud_color = []

        # Convert depth to 3D points
        for i, point_cloud_id in enumerate(results_dict["point_cloud_ids"].tolist()):
            (
                assigned_colmap_frameID,
                assigned_keypoint_index,
            ) = self.point_cloud_assigned_imgID_kptID[point_cloud_id]

            keypoint = self.colmap_images[assigned_colmap_frameID].xys[
                assigned_keypoint_index
            ][
                None
            ]  # 1*2
            intrinsic = self.colmap_frame_dict[assigned_colmap_frameID]["intrinsic"]
            pose = [
                qvec2rotmat(self.colmap_images[assigned_colmap_frameID].qvec),
                self.colmap_images[assigned_colmap_frameID].tvec,
            ]  # [R: 3*3, t: 3]
            T = convert_pose2T(pose)  # 4*4
            pose_to_world = convert_T2pose(np.linalg.inv(T))  # [R: 3*3, t: 3]

            # unproject keypoints to world space
            kpt_h = (
                np.concatenate([keypoint, np.ones((keypoint.shape[0], 1))], axis=-1)
                * results_dict["depth"][i]
            ).T  # 3*1

            kpt_cam = np.linalg.inv(intrinsic) @ kpt_h
            kpt_world = pose_to_world[0] @ kpt_cam + pose_to_world[1][:, None]  # 3*1

            # update pointcloud coord
            self.colmap_3ds[point_cloud_id] = self.colmap_3ds[point_cloud_id]._replace(
                xyz=kpt_world.squeeze(-1)
            )  # namedtuple format attribute setting

            point_cloud.append(self.colmap_3ds[point_cloud_id].xyz)
            point_cloud_color.append(self.colmap_3ds[point_cloud_id].rgb)

        # Update colmap image keypoints:
        for colmap_frame_id, colmap_image in self.colmap_images.items():
            keypoints = colmap_image.xys
            point3D_ids = colmap_image.point3D_ids
            registrated_mask = point3D_ids != -1

            keypoints_masked = keypoints[registrated_mask]
            point3D_ids_masked = point3D_ids[registrated_mask]

            if point3D_ids_masked.shape[0] == 0:
                # No keypoints registrated scenario
                continue

            # Get corresponding 3D coordinates
            point3D_coords = []
            for point3D_id in point3D_ids_masked.tolist():
                point3D = self.colmap_3ds[point3D_id].xyz
                point3D_coords.append(point3D)
            point3D_coords = np.stack(point3D_coords, axis=0)  # N*3

            # Get frame pose:
            intrinsic = self.colmap_frame_dict[colmap_frame_id]["intrinsic"]
            pose = [
                qvec2rotmat(colmap_image.qvec),
                colmap_image.tvec,
            ]  # [R: 3*3, t: 3]

            # Project 3D points to frame
            kpts_cam = pose[0] @ point3D_coords.T + pose[1][:, None] # 3*N
            kpts_frame_h = (intrinsic @ kpts_cam).T # N*3
            projected_kpts = kpts_frame_h[:, :2] / (kpts_frame_h[:, [2]] + 1e-4) # N*2

            # Update to colmap frames
            keypoints[registrated_mask, :] = projected_kpts
            self.colmap_images[colmap_frame_id] = colmap_image._replace(xys=keypoints)

        # Write results to colmap file format
        os.makedirs(self.refined_save_dir, exist_ok=True)
        write_model(
            self.colmap_cameras,
            self.colmap_images,
            self.colmap_3ds,
            self.refined_save_dir,
            ext=".bin",
        )
        write_model(
            self.colmap_cameras,
            self.colmap_images,
            self.colmap_3ds,
            self.refined_save_dir,
            ext=".txt",
        )

        # visualize
        if visualize and self.vis_path is not None:
            T_all = np.stack(T_all)  # N*4*4
            point_cloud = np.stack(point_cloud)  # N*3
            point_cloud_color = np.stack(point_cloud_color)  # N*3

            old_point_cloud = np.stack(old_point_cloud)  # N*3
            old_point_cloud_color = np.stack(old_point_cloud_color)  # N*3
            old_pose = np.stack(old_pose)  # N*4*4

            colmap_vis3d_dump_dir, name = self.vis_path.rsplit("/", 1)

            vis_cameras_point_clouds(
                point_cloud,
                colmap_vis3d_dump_dir,
                name,
                point_cloud_color,
                old_point_cloud,
                old_point_cloud_color,
            )

    def _is_colmap_3d_empty(self, colmap_3d_state):
        num_non_empty = 0
        for state in colmap_3d_state.values():
            if len(state) == 1:
                num_non_empty += 1

        return num_non_empty == 0

    def _sort_colmap_images_state(self, colmap_images_state):
        # Sort colmap images state by "unoccupied_num"
        colmap_images_state_sorted = {
            k: v
            for k, v in sorted(
                colmap_images_state.items(),
                key=lambda item: item[1]["unoccupied_num"],
                reverse=True,
            )
        }
        return colmap_images_state_sorted

    def _update_colmap_images_state_unoccupied_number(self, colmap_images_state):
        # Update colmap image state's occupied
        for key in colmap_images_state.keys():
            colmap_images_state[key]["unoccupied_num"] = (
                colmap_images_state[key]["state"] == -2
            ).sum()
        return colmap_images_state

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        img_name = self.img_list[idx]
        img_scale = read_grayscale(
            img_name, (self.img_resize,), df=self.df, ret_scales=True,
        )
        img, scale = map(lambda x: x[None], img_scale)  # no dataloader operation
        data = {
            "image": img,  # 1*1*H*W because no dataloader operation, if batch: 1*H*W
            "scale": scale,  # 1*2
            "f_name": img_name,
            "img_name": img_name,
            "frameID": idx,
            "img_path": [img_name],
        }
        return data

