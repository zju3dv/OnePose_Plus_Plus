import argparse
from shutil import copyfile
import os
import os.path as osp
from shutil import rmtree
import numpy as np
import cv2
from glob import glob
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from src.utils.data_utils import get_image_crop_resize, get_K_crop_resize

id2name_dict = {
    1: "ape",
    2: "benchvise",
    4: "camera",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input:
    parser.add_argument(
        "--data_base_dir", type=str, default="data/LINEMOD"
    )
    parser.add_argument("--obj_id", type=str, default="1")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--assign_onepose_id", type=str, default="0801")
    parser.add_argument("--add_detector_noise", action='store_true')
    parser.add_argument("--use_yolo_box", action='store_true')
    parser.add_argument("--yolo_box_base_path", default="data/LINEMOD/yolo_detection")

    # Output:
    parser.add_argument(
        "--output_data_dir", type=str, default="data/datasets/LM_dataset"
    )

    args = parser.parse_args()
    return args

def parse_models_info_txt(models_info_txt_path):
    models_info_dict = {}
    with open(models_info_txt_path, "r") as f:
        txt_list = f.readlines()
        for obj_info in txt_list:
            obj_info_splited = obj_info.split(" ")
            obj_id = obj_info_splited.pop(0)
            model_info = {}
            for id in range(0, len(obj_info_splited), 2):
                model_info[obj_info_splited[id]] = float(obj_info_splited[id + 1])
            models_info_dict[obj_id] = model_info
    return models_info_dict


if __name__ == "__main__":
    args = parse_args()
    obj_name = id2name_dict[int(args.obj_id)]

    logger.info(f"Working on obj:{obj_name}")

    image_seq_dir = osp.join(
        args.data_base_dir,
        "real_train" if args.split == "train" else "real_test",
        obj_name,
    )
    model_path = osp.join(args.data_base_dir, "models", obj_name, obj_name + ".ply")
    models_info_dict = parse_models_info_txt(
        osp.join(args.data_base_dir, "models", "models_info.txt")
    )
    assert osp.exists(image_seq_dir)

    rgb_pths = glob(os.path.join(image_seq_dir, "*-color.png"))

    # Construct output data file structure
    output_data_base_dir = args.output_data_dir
    obj_full_name = "-".join([args.assign_onepose_id, "lm" + str(int(args.obj_id)), "others"])
    output_data_obj_dir = osp.join(
        output_data_base_dir,
        obj_full_name,
    )
    if (not args.add_detector_noise) and (not args.use_yolo_box):
        sequence_name = "-".join(
            ["lm" + str(int(args.obj_id)), "1" if args.split == "train" else "2"]
        )  # label seq 0 for mapping data, label seq 1 for test data
    else:
        sequence_name = "-".join(
            ["lm" + str(int(args.obj_id)), "3"]
        )  # label seq 0 for mapping data, label seq 2 for test data with detection inferenced by YOLOv5
    output_data_seq_dir = osp.join(output_data_obj_dir, sequence_name,)
    if osp.exists(output_data_seq_dir):
        rmtree(output_data_seq_dir)
    Path(output_data_seq_dir).mkdir(parents=True, exist_ok=True)

    color_path = osp.join(output_data_seq_dir, "color")
    color_full_path = osp.join(output_data_seq_dir, "color_full")
    intrin_path = osp.join(output_data_seq_dir, "intrin_ba")
    intrin_origin_path = osp.join(output_data_seq_dir, "intrin")
    poses_path = osp.join(output_data_seq_dir, "poses_ba")
    Path(color_path).mkdir(exist_ok=True)
    Path(color_full_path).mkdir(exist_ok=True)
    Path(intrin_path).mkdir(exist_ok=True)
    Path(intrin_origin_path).mkdir(exist_ok=True)
    Path(poses_path).mkdir(exist_ok=True)

    # Save model info:
    if args.split == "train":
        model_min_xyz = np.array(
            [
                models_info_dict[str(int(args.obj_id))]["min_x"],
                models_info_dict[str(int(args.obj_id))]["min_y"],
                models_info_dict[str(int(args.obj_id))]["min_z"],
            ]
        )
        model_size_xyz = np.array(
            [
                models_info_dict[str(int(args.obj_id))]["size_x"],
                models_info_dict[str(int(args.obj_id))]["size_y"],
                models_info_dict[str(int(args.obj_id))]["size_z"],
            ]
        )
        scale = model_size_xyz / 1000 # convert to m

        # Save 3D bbox:
        corner_in_cano = np.array([
            [-scale[0], -scale[0], -scale[0], -scale[0],  scale[0],  scale[0],  scale[0],  scale[0]],
            [-scale[1], -scale[1],  scale[1],  scale[1], -scale[1], -scale[1],  scale[1],  scale[1]],
            [-scale[2],  scale[2],  scale[2], -scale[2], -scale[2],  scale[2],  scale[2], -scale[2]],
        ]).T
        corner_in_cano = corner_in_cano[:, :3] * 0.5
        np.savetxt(osp.join(output_data_obj_dir, "box3d_corners.txt"), corner_in_cano) # 8*3

        # Copy eval model and save diameter:
        model_eval_path = model_path
        diameter = models_info_dict[str(int(args.obj_id))]["diameter"] / 1000  # convert to m
        assert osp.exists(
            model_eval_path
        ), f"model eval path:{model_eval_path} not exists!"
        copyfile(model_eval_path, osp.join(output_data_obj_dir, "model_eval.ply")) # NOTE: models' units are m
        np.savetxt(osp.join(output_data_obj_dir, "diameter.txt"), np.array([diameter]))


    img_id = 0
    for global_id, image_path in tqdm(enumerate(rgb_pths), total=len(rgb_pths)):
        dataset_img_id, file_label = (
            osp.splitext(image_path)[0].rsplit("/", 1)[1].split("-")
        )
        img_ext = osp.splitext(image_path)[1]
        K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
        pose = np.loadtxt(
            osp.join(image_seq_dir, "-".join([dataset_img_id, "pose"]) + ".txt")
        )
        original_img = cv2.imread(image_path)
        img_h,img_w = original_img.shape[:2]

        if args.split == 'train':
            # Load GT box directly:
            x0, y0, w, h = (
                np.loadtxt(
                    osp.join(image_seq_dir, "-".join([dataset_img_id, "box"]) + ".txt")
                )
                .astype(np.int)
                .tolist()
            )
            x1, y1 = x0 + w, y0 + h

        else:
            if args.use_yolo_box:
                yolo_box_base_path = args.yolo_box_base_path
                yolo_box_obj_path = osp.join(yolo_box_base_path, args.split, obj_full_name, 'labels')
                yolo_box = np.loadtxt(osp.join(yolo_box_obj_path, dataset_img_id+'.txt'))
                assert yolo_box.shape[0] != 0, f"img id:{dataset_img_id} no box detected!"
                if len(yolo_box.shape) == 2:
                    want_id = np.argsort(yolo_box[:,5])[0]
                    yolo_box = yolo_box[want_id]
                
                x_c_n, y_c_n, w_n, h_n = yolo_box[1:5]
                x0_n, y0_n = x_c_n - w_n / 2, y_c_n - h_n /2

                x0, y0, w, h = int(x0_n * img_w), int(y0_n * img_h), int(w_n * img_w), int(h_n * img_h)
                x1, y1 = x0 + w, y0 + h

            else:
                # Use GT box
                x0, y0, w, h = (
                    np.loadtxt(
                        osp.join(image_seq_dir, "-".join([dataset_img_id, "box"]) + ".txt")
                    )
                    .astype(np.int)
                    .tolist()
                )
                x1, y1 = x0 + w, y0 + h


        if not args.add_detector_noise:
            compact_percent = 0.3
            x0 -= int(w * compact_percent)
            y0 -= int(h * compact_percent)
            x1 += int(w * compact_percent)
            y1 += int(h * compact_percent)
        else:
            compact_percent = 0.3
            offset_percent = np.random.uniform(low=-1*compact_percent, high=1*compact_percent)
            # apply compact noise:
            x0 -= int(w * compact_percent)
            y0 -= int(h * compact_percent)
            x1 += int(w * compact_percent)
            y1 += int(h * compact_percent)
            # apply offset noise:
            x0 += int(w * offset_percent)
            y0 += int(h * offset_percent)
            x1 += int(w * offset_percent)
            y1 += int(h * offset_percent)

        # Crop image by 2D visible bbox, and change K
        box = np.array([x0, y0, x1, y1])
        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop, K_crop_homo = get_K_crop_resize(box, K, resize_shape)
        image_crop, _ = get_image_crop_resize(original_img, box, resize_shape)

        box_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([256, 256])
        K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
        image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)

        # Save to aim dir:
        cv2.imwrite(osp.join(color_path, str(global_id) + img_ext), image_crop)
        cv2.imwrite(osp.join(color_full_path, str(global_id) + img_ext), original_img)
        np.savetxt(osp.join(intrin_path, str(global_id) + ".txt"), K_crop)
        np.savetxt(osp.join(intrin_origin_path, str(global_id) + ".txt"), K) # NOTE: intrinsic of full image. Used to eval Proj2D metric
        np.savetxt(osp.join(poses_path, str(global_id) + ".txt"), pose)
