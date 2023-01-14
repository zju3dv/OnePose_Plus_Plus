import os.path as osp
import cv2
import torch
import numpy as np
import natsort
import pytorch_lightning as pl

from src.KeypointFreeSfM.loftr_for_sfm import LoFTR_for_OnePose_Plus, default_cfg
from src.utils.colmap.read_write_model import read_model
from src.utils.data_utils import get_K_crop_resize, get_image_crop_resize
from src.utils.vis_utils import reproj

cfgs = {
    "model": {
        "method": "LoFTR",
        "weight_path": "weight/LoFTR_wsize9.ckpt",
        "seed": 666,
    },
}

def build_2D_match_model(args):
    pl.seed_everything(args['seed'])

    if args['method'] == 'LoFTR':
        matcher = LoFTR_for_OnePose_Plus(config=default_cfg)
        # load checkpoints
        state_dict = torch.load(args['weight_path'], map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k.replace("matcher.", "")] = state_dict.pop(k)
        matcher.load_state_dict(state_dict, strict=True)
        matcher.eval()
    else:
        raise NotImplementedError

    return matcher

class LocalFeatureObjectDetector():
    def __init__(self, sfm_ws_dir, n_ref_view=15, output_results=False, detect_save_dir=None, K_crop_save_dir=None):
        matcher = build_2D_match_model(cfgs['model']) 
        self.matcher = matcher.cuda()
        self.db_imgs, self.db_corners_homo = self.load_ref_view_images(sfm_ws_dir, n_ref_view)
        self.output_results = output_results
        self.detect_save_dir = detect_save_dir
        self.K_crop_save_dir = K_crop_save_dir

    def load_ref_view_images(self, sfm_ws_dir, n_ref_view):
        assert osp.exists(sfm_ws_dir), f"SfM work space:{sfm_ws_dir} not exists!"
        cameras, images, points3D = read_model(sfm_ws_dir)
        idx = 0
        sample_gap = len(images) // n_ref_view
        db_image_paths = natsort.natsorted([image.name for image in images.values()])

        db_imgs = []  # id: image
        db_corners_homo = []
        for idx in range(1, len(images), sample_gap):
            db_img_path = db_image_paths[idx]

            db_img = cv2.imread(db_img_path, cv2.IMREAD_GRAYSCALE)
            db_imgs.append(torch.from_numpy(db_img)[None][None] / 255.0)
            H, W = db_img.shape[-2:]
            db_corners_homo.append(
                np.array(
                    [
                        [0, 0, 1],
                        [W, 0, 1], # w, 0
                        [0, H, 1], # 0, h
                        [W, H, 1],
                    ]
                ).T  # 3*4
            )

        return db_imgs, db_corners_homo

    @torch.no_grad()
    def match_worker(self, query):
        detect_results_dict = {}
        for idx, db_img in enumerate(self.db_imgs):

            match_data = {"image0": db_img.cuda(), "image1": query.cuda()}
            self.matcher(match_data)
            mkpts0 = match_data["mkpts0_f"].cpu().numpy()
            mkpts1 = match_data["mkpts1_f"].cpu().numpy()

            if mkpts0.shape[0] < 6:
                affine = None
                inliers = np.empty((0))
                img_center = (query.shape[-1] // 2, query.shape[-2] // 2)
                detect_results_dict[idx] = {
                    "inliers": inliers,
                    "bbox": np.array([img_center[0] - 500, img_center[1] - 500, img_center[0] + 500, img_center[1] + 500]) # [w,h]
                }
                continue

            affine, inliers = cv2.estimateAffine2D(
                mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=6
            )

            # Estimate box:
            four_corner = self.db_corners_homo[idx]

            bbox = (affine @ four_corner).T.astype(np.int32)  # 4*2

            left_top = np.min(bbox, axis=0)
            right_bottom = np.max(bbox, axis=0)

            w, h = right_bottom - left_top
            offset_percent = 0.0
            x0 = left_top[0] - int(w * offset_percent)
            y0 = left_top[1] - int(h * offset_percent)
            x1 = right_bottom[0] + int(w * offset_percent)
            y1 = right_bottom[1] + int(h * offset_percent)

            detect_results_dict[idx] = {
                "inliers": inliers,
                "bbox": np.array([x0, y0, x1, y1]),
            }
        return detect_results_dict

    def detect_by_matching(self, query):
        detect_results_dict = self.match_worker(query)

        # Sort multiple bbox candidate and use bbox with maxium inliers:
        idx_sorted = [
            k
            for k, _ in sorted(
                detect_results_dict.items(),
                reverse=True,
                key=lambda item: item[1]["inliers"].sum(),
            )
        ]
        return detect_results_dict[idx_sorted[0]]["bbox"]

    def crop_img_by_bbox(self, query_img_path, bbox, K=None, crop_size=512):
        """
        Crop image by detect bbox
        Input:
            query_img_path: str,
            bbox: np.ndarray[x0, y0, x1, y1],
            K[optional]: 3*3
        Output:
            image_crop: np.ndarray[crop_size * crop_size],
            K_crop[optional]: 3*3
        """
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = bbox[2], bbox[3]
        origin_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)

        resize_shape = np.array([y1 - y0, x1 - x0])
        if K is not None:
            K_crop, K_crop_homo = get_K_crop_resize(bbox, K, resize_shape)
        image_crop, trans1 = get_image_crop_resize(origin_img, bbox, resize_shape)

        bbox_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([crop_size, crop_size])
        if K is not None:
            K_crop, K_crop_homo = get_K_crop_resize(bbox_new, K_crop, resize_shape)
        image_crop, trans2 = get_image_crop_resize(image_crop, bbox_new, resize_shape)
        
        return image_crop, K_crop if K is not None else None
    
    def save_detection(self, crop_img, query_img_path):
        if self.output_results and self.detect_save_dir is not None:
            cv2.imwrite(osp.join(self.detect_save_dir, osp.basename(query_img_path)), crop_img)
    
    def save_K_crop(self, K_crop, query_img_path):
        if self.output_results and self.K_crop_save_dir is not None:
            np.savetxt(osp.join(self.K_crop_save_dir, osp.splitext(osp.basename(query_img_path))[0] + '.txt'), K_crop) # K_crop: 3*3

    def detect(self, query_img, query_img_path, K, crop_size=512):
        """
        Detect object by local feature matching and crop image.
        Input:
            query_image: np.ndarray[1*1*H*W],
            query_img_path: str,
            K: np.ndarray[3*3], intrinsic matrix of original image
        Output:
            bounding_box: np.ndarray[x0, y0, x1, y1]
            cropped_image: torch.tensor[1 * 1 * crop_size * crop_size] (normalized),
            cropped_K: np.ndarray[3*3];
        """
        if len(query_img.shape) != 4:
            query_inp = query_img[None].cuda()
        else:
            query_inp = query_img.cuda()
        
        # Detect bbox and crop image:
        bbox = self.detect_by_matching(
            query=query_inp,
        )
        image_crop, K_crop = self.crop_img_by_bbox(query_img_path, bbox, K, crop_size=crop_size)
        self.save_detection(image_crop, query_img_path)
        self.save_K_crop(K_crop, query_img_path)

        # To Tensor:
        image_crop = image_crop.astype(np.float32) / 255
        image_crop_tensor = torch.from_numpy(image_crop)[None][None].cuda()

        return bbox, image_crop_tensor, K_crop
    
    def previous_pose_detect(self, query_img_path, K, pre_pose, bbox3D_corner, crop_size=512):
        """
        Detect object by projecting 3D bbox with estimated last frame pose.
        Input:
            query_image_path: str,
            K: np.ndarray[3*3], intrinsic matrix of original image
            pre_pose: np.ndarray[3*4] or [4*4], pose of last frame
            bbox3D_corner: np.ndarray[8*3], corner coordinate of annotated 3D bbox
        Output:
            bounding_box: np.ndarray[x0, y0, x1, y1]
            cropped_image: torch.tensor[1 * 1 * crop_size * crop_size] (normalized),
            cropped_K: np.ndarray[3*3];
        """
        # Project 3D bbox:
        proj_2D_coor = reproj(K, pre_pose, bbox3D_corner)
        x0, y0 = np.min(proj_2D_coor, axis=0)
        x1, y1 = np.max(proj_2D_coor, axis=0)
        bbox = np.array([x0, y0, x1, y1]).astype(np.int32)

        image_crop, K_crop = self.crop_img_by_bbox(query_img_path, bbox, K, crop_size=crop_size)
        self.save_detection(image_crop, query_img_path)
        self.save_K_crop(K_crop, query_img_path)

        # To Tensor:
        image_crop = image_crop.astype(np.float32) / 255
        image_crop_tensor = torch.from_numpy(image_crop)[None][None].cuda()

        return bbox, image_crop_tensor, K_crop