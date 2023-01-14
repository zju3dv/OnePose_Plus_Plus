import cv2
import logging
import os.path as osp
from loguru import logger
import numpy as np

from pathlib import Path
from src.utils.colmap.read_write_model import Camera, Image
from src.utils.colmap.read_write_model import rotmat2qvec
from src.utils.colmap.read_write_model import write_model
from src.utils.data_utils import get_K


def get_pose_from_txt(img_index, pose_dir):
    """ Read 4x4 transformation matrix from txt """
    pose_file = osp.join(pose_dir, '{}.txt'.format(img_index))
    pose = np.loadtxt(pose_file)
    
    tvec = pose[:3, 3].reshape(3, )
    qvec = rotmat2qvec(pose[:3, :3]).reshape(4, )
    return pose, tvec, qvec


def get_intrin_from_txt(img_index, intrin_dir):
    """ Read 3x3 intrinsic matrix from txt """
    intrin_file = osp.join(intrin_dir, '{}.txt'.format(img_index))
    intrin = np.loadtxt(intrin_file)
    
    return intrin


def import_data(img_lists, do_ba=False):
    """ Import intrinsics and camera pose info """
    points3D_out = {}
    images_out = {}
    cameras_out = {}

    def compare(img_name):
        key = img_name.split('/')[-1]
        return int(key.split('.')[0])
    img_lists.sort(key=compare)

    key, img_id, camera_id = 0, 0, 0
    xys_ = np.zeros((0, 2), float) 
    point3D_ids_ = np.full(0, -1, int) # will be filled after triangulation 

    # import data
    # suppose the image_path can be formatted as  "/path/.../color/***.png"
    img_type = img_lists[0].split('/')[-2]
    for img_path in img_lists:
        key += 1
        img_id += 1
        camera_id += 1
        
        img_name = img_path.split('/')[-1]
        # base_dir = osp.dirname(img_path).rstrip('color') # root dir of this sequence
        base_dir = osp.dirname(osp.dirname(img_path))
        img_index = int(img_name.split('.')[0])
        
        # read pose
        if do_ba:
            pose_dir = osp.join(base_dir, 'poses')
        else:
            pose_dir = osp.join(base_dir, 'poses_ba')
            if not osp.exists(pose_dir):
                logger.warning(f"pose dir :{pose_dir} not exists, use poses instead!")
                pose_dir = osp.join(base_dir, 'poses')

        _, tvec, qvec = get_pose_from_txt(img_index, pose_dir)

        # read intrinsic
        if img_type == 'color' or img_type == 'color':
            if do_ba:
                intrin_dir = osp.join(base_dir, 'intrin')
            else:
                intrin_dir = osp.join(base_dir, 'intrin_ba')
                if not osp.exists(intrin_dir):
                    logger.warning(f"intrin dir :{intrin_dir} not exists, use 'intrin' instead!")
                    intrin_dir = osp.join(base_dir, 'intrin')

            K = get_intrin_from_txt(img_index, intrin_dir)
            fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]
        else:
            raise NotImplementedError
            
        image = cv2.imread(img_path)
        h, w, _ = image.shape
        
        image = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=img_path,
            xys=xys_,
            point3D_ids=point3D_ids_
        )
        
        camera = Camera(
            id=camera_id,
            model='PINHOLE',
            width=w,
            height=h,
            params=np.array([fx, fy, cx, cy])
        )
        
        images_out[key] = image
        cameras_out[key] = camera
    
    return cameras_out, images_out, points3D_out


def generate_model(img_lists, empty_dir, do_ba=False):
    """ Write intrinsics and camera poses into COLMAP format model"""
    logging.info('Generate empty model...')
    model = import_data(img_lists, do_ba)

    logging.info(f'Writing the COLMAP model to {empty_dir}')
    Path(empty_dir).mkdir(exist_ok=True, parents=True)
    write_model(*model, path=str(empty_dir), ext='.bin')
    logging.info('Finishing writing model.')
    