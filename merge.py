import hydra
import json
import os

import os.path as osp

from pathlib import Path
from loguru import logger

from src.utils.path_utils import get_test_seq_path, get_gt_pose_path_by_color


def merge_train_core(
    anno_2d_file,
    avg_anno_3d_file,
    idxs_file,
    img_id,
    ann_id,
    images,
    annotations,
):
    """ Merge training annotations of different objects"""

    with open(anno_2d_file, "r") as f:
        annos_2d = json.load(f)

    for anno_2d in annos_2d:
        img_id += 1
        info = {
            "id": img_id,
            "img_file": anno_2d["img_file"],
        }
        images.append(info)

        ann_id += 1
        anno = {
            "image_id": img_id,
            "id": ann_id,
            "pose_file": anno_2d["pose_file"],
            "anno2d_file": anno_2d["anno_file"],
            "avg_anno3d_file": avg_anno_3d_file,
            "idxs_file": idxs_file,
        }
        annotations.append(anno)

    return img_id, ann_id


def merge_val_core(
    data_dir,
    name,
    avg_anno_3d_file,
    idxs_file,
    img_id,
    ann_id,
    images,
    annotations,
    last_n_seq_as_test=1,
    downsample=5,
):
    """ Merge validation annotaions of different objects"""
    obj_root = osp.join(data_dir, name)
    test_seq_paths = get_test_seq_path(obj_root, last_n_seq_as_test=last_n_seq_as_test)

    for test_seq_path in test_seq_paths:
        color_dir = osp.join(test_seq_path, "color")
        img_names = os.listdir(color_dir)

        for img_name in img_names[::downsample]:
            img_file = osp.join(color_dir, img_name)

            img_id += 1
            info = {"id": img_id, "img_file": img_file}
            images.append(info)

            ann_id += 1
            anno = {
                "image_id": img_id,
                "id": ann_id,
                "pose_file": get_gt_pose_path_by_color(img_file),
                "avg_anno3d_file": avg_anno_3d_file,
                "idxs_file": idxs_file,
            }
            annotations.append(anno)

    return img_id, ann_id


def merge_(cfg, names, split):
    data_dir = cfg.datamodule.data_dir
    sfm_dir = cfg.datamodule.sfm_dir

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    all_data_names = os.listdir(
        osp.join(
            sfm_dir,
            f"outputs_{cfg.match_type}_{cfg.network.detection}_{cfg.network.matching}",
        )
    )
    id2datafullname = {
        data_name[:4]: data_name for data_name in all_data_names if "-" in data_name
    }
    for name in names:
        if len(name) == 4:
            # ID only!
            if name in id2datafullname:
                name = id2datafullname[name]
            else:
                logger.warning(f"id {name} not exist in sfm directory")
        anno_dir = osp.join(
            sfm_dir,
            f"outputs_{cfg.match_type}_{cfg.network.detection}_{cfg.network.matching}",
            name,
            "anno",
        )

        logger.info(f"Merging anno dir: {anno_dir}")
        anno_2d_file = osp.join(anno_dir, "anno_2d.json")
        avg_anno_3d_file = osp.join(anno_dir, "anno_3d_average.npz")
        idxs_file = osp.join(anno_dir, "idxs.npy")

        if not osp.isfile(anno_2d_file) or not osp.isfile(avg_anno_3d_file):
            logger.info(f"No annotation in: {anno_dir}")
            continue

        if split == "train":
            img_id, ann_id = merge_train_core(
                anno_2d_file,
                avg_anno_3d_file,
                idxs_file,
                img_id,
                ann_id,
                images,
                annotations,
            )
        elif split == "val":
            img_id, ann_id = merge_val_core(
                data_dir,
                name,
                avg_anno_3d_file,
                idxs_file,
                img_id,
                ann_id,
                images,
                annotations,
                last_n_seq_as_test=cfg.val_use_last_n_seq,
                downsample=1,
            )
        else:
            raise NotImplementedError

    logger.info(f"Total num for {split}: {len(images)}")
    instances = {"images": images, "annotations": annotations}

    out_path = cfg.datamodule.out_path.format(split)
    out_dir = osp.dirname(cfg.datamodule.out_path)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(instances, f)


def merge_anno(cfg):
    # Parse names
    names = cfg.names

    if isinstance(names, str):
        # Parse object directory
        assert isinstance(names, str)
        exception_obj_name_list = cfg.exception_obj_names
        top_k_obj = cfg.top_k_obj
        logger.info(f"Process all objects in directory:{names}")

        object_names = []
        object_names_list = os.listdir(names)[:top_k_obj]
        for object_name in object_names_list:
            if "-" not in object_name:
                continue
            if object_name in exception_obj_name_list:
                continue
            object_names.append(object_name)

        names = object_names

    merge_(cfg, cfg.names, split=cfg.split)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
