import os
import os.path as osp
from loguru import logger
from shutil import copyfile, rmtree
import numpy as np
import subprocess
import multiprocessing

from src.utils.data_io import load_h5

def colmapRunner(
    img_list,
    img_pairs,
    work_dir,
    feature_out,
    match_out,
    colmap_coarse_dir,
    colmap_debug=True,
    colmap_verbose=True,
    colmap_configs=None,
):
    # Build Colmap file path
    base_path = work_dir
    os.makedirs(base_path, exist_ok=True)
    colmap_temp_path = osp.join(base_path, "temp_output")
    colmap_output_path = colmap_coarse_dir
    # create temp directory
    if osp.exists(colmap_temp_path):
        logger.info(" -- temp path exists - cleaning up from crash")
        rmtree(colmap_temp_path)
        if os.path.exists(colmap_output_path):
            rmtree(colmap_output_path)

    # create output directory
    if osp.exists(colmap_output_path):
        if not colmap_debug:
            logger.info("colmap results already exists, don't need to run colmap")
            return
        else:
            rmtree(colmap_output_path)

    os.makedirs(colmap_temp_path)
    os.makedirs(colmap_output_path)

    # Load keypoints and matches:
    keypoints_dict = load_h5(feature_out)
    matches_dict = load_h5(match_out)

    # Create colmap-friendy structures
    os.makedirs(os.path.join(colmap_temp_path, "images"))
    os.makedirs(os.path.join(colmap_temp_path, "features"))
    img_paths = img_list

    # Copy images to colmap friendly format:
    for _src in img_paths:
        _dst = osp.join(colmap_temp_path, "images", osp.basename(_src))
        copyfile(_src, _dst)
    logger.info(f"Image copy finish! Copy {len(img_paths)} images!")

    num_kpts = []
    # Write features to colmap friendly format
    for img_path in img_paths:
        img_name = osp.basename(img_path)
        # load keypoints:
        keypoints = keypoints_dict[img_name]
        # kpts file to write to:
        kp_file = osp.join(colmap_temp_path, "features", img_name + ".txt")
        num_kpts.append(keypoints.shape[0])
        # open file to write
        with open(kp_file, "w") as f:
            # Retieve the number of keypoints
            len_keypoints = len(keypoints)
            f.write(str(len_keypoints) + " " + str(128) + "\n")
            for i in range(len_keypoints):
                kp = " ".join(str(k) for k in keypoints[i][:4])
                desc = " ".join(str(0) for d in range(128))
                f.write(kp + " " + desc + "\n")
    logger.info(
        f"Feature format convert finish! Converted {len(img_paths)} images, have: {np.array(num_kpts)} keypoints"
    )

    # Write matches to colmap friendly format:
    match_file = os.path.join(colmap_temp_path, "matches.txt")
    num_matches = []
    with open(match_file, "w") as f:
        for i, img_pair in enumerate(img_pairs):
            img0_path, img1_path = img_pair.split(" ")
            img0_name = osp.basename(img0_path)
            img1_name = osp.basename(img1_path)

            # Load matches
            key = " ".join([img0_name, img1_name])
            matches = np.squeeze(matches_dict[key])
            # only write when matches are given
            if matches.ndim == 2:
                num_matches.append(matches.shape[1])
                f.write(img0_name + " " + img1_name + "\n")
                for _i in range(matches.shape[1]):
                    f.write(str(matches[0, _i]) + " " + str(matches[1, _i]) + "\n")
                f.write("\n")
            else:
                num_matches.append(0)
    logger.info(
        f"Match format convert finish, Converted {len(img_pairs)} pairs, min match: {np.array(num_matches).min()}, max match: {np.array(num_matches).max()}"
    )

    try:
        print("COLMAP Feature Import")
        cmd = ["colmap", "feature_importer"]
        cmd += ["--database_path", os.path.join(colmap_output_path, "databases.db")]
        cmd += ["--image_path", os.path.join(colmap_temp_path, "images")]
        cmd += ["--import_path", os.path.join(colmap_temp_path, "features")]
        colmap_res = subprocess.run(cmd)

        if colmap_res.returncode != 0:
            raise RuntimeError(" -- COLMAP failed to import features!")

        print("COLMAP Match Import")
        cmd = ["colmap", "matches_importer"]
        cmd += ["--database_path", os.path.join(colmap_output_path, "databases.db")]
        cmd += ["--match_list_path", os.path.join(colmap_temp_path, "matches.txt")]
        cmd += ["--match_type", "raw"]
        cmd += ["--SiftMatching.use_gpu", "0"]
        colmap_res = subprocess.run(cmd)
        if colmap_res.returncode != 0:
            raise RuntimeError(" -- COLMAP failed to import matches!")

        print("COLMAP Mapper")
        cmd = ["colmap", "mapper"]
        cmd += ["--image_path", os.path.join(colmap_temp_path, "images")]
        cmd += ["--database_path", os.path.join(colmap_output_path, "databases.db")]
        cmd += ["--output_path", colmap_output_path]
        if colmap_configs is not None and "min_model_size" in colmap_configs:
            cmd += ["--Mapper.min_model_size", str(colmap_configs["min_model_size"])]
        else:
            cmd += ["--Mapper.min_model_size", str(6)]
        cmd += ["--Mapper.num_threads", str(min(multiprocessing.cpu_count(), 32))]

        if colmap_configs is not None and "filter_max_reproj_error" in colmap_configs:
            cmd += [
                "--Mapper.filter_max_reproj_error",
                str(colmap_configs["filter_max_reproj_error"]),
            ]
        else:
            cmd += [
                "--Mapper.filter_max_reproj_error",
                str(4),
            ]

        if colmap_verbose:
            colmap_res = subprocess.run(cmd)
        else:
            colmap_res = subprocess.run(cmd, capture_output=True)
            with open(osp.join(colmap_output_path, "output.txt"), "w") as f:
                f.write(colmap_res.stdout.decode())

        if colmap_res.returncode != 0:
            print("warning! colmap failed to run mapper!")

        rmtree(colmap_temp_path)

    except Exception as err:
        rmtree(colmap_temp_path)
        rmtree(colmap_output_path)

        # Re-throw error
        print(err)
        raise RuntimeError("Parts of colmap runs returns failed state!")