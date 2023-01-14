import os
import h5py
import logging
import tqdm
import subprocess
import os.path as osp
import numpy as np
import ray

from pathlib import Path
from src.utils.colmap.read_write_model import CAMERA_MODEL_NAMES, Image, read_cameras_binary, read_images_binary
from src.utils.colmap.database import COLMAPDatabase


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def geometric_verification(colmap_path, database_path, pairs_path, verbose=True):
    """ Geometric verfication """
    logging.info('Performing geometric verification of the matches...')
    cmd = [
        str(colmap_path), 'matches_importer',
        '--database_path', str(database_path),
        '--match_list_path', str(pairs_path),
        '--match_type', 'pairs'
    ]
    if verbose:
        ret = subprocess.call(cmd)
    else:
        ret_all = subprocess.run(cmd, capture_output=True)
        ret = ret_all.returncode
    if ret != 0:
        logging.warning('Problem with matches_importer, existing.')
        exit(ret)


def create_db_from_model(empty_model, database_path):
    """ Create COLMAP database file from empty COLMAP binary file. """
    if database_path.exists():
        logging.warning('Database already exists.')
    
    cameras = read_cameras_binary(str(empty_model / 'cameras.bin'))
    images = read_images_binary(str(empty_model / 'images.bin'))

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    
    for i, camera in cameras.items():
        model_id = CAMERA_MODEL_NAMES[camera.model].model_id
        db.add_camera(model_id, camera.width, camera.height, camera.params,
                      camera_id=i, prior_focal_length=True)
    
    for i, image in images.items():
        db.add_image(image.name, image.camera_id, image_id=i)
    
    db.commit()
    db.close()
    return {image.name: i for i, image in images.items()}


def import_features(image_ids, database_path, feature_path, verbose=True):
    """ Import keypoints info into COLMAP database. """
    logging.info("Importing features into the database...")
    feature_file = h5py.File(str(feature_path), 'r')
    db = COLMAPDatabase.connect(database_path)
    
    if verbose:
        iter_obj = tqdm.tqdm(image_ids.items())
    else:
        iter_obj = image_ids.items()
    for image_name, image_id in iter_obj:
        keypoints = feature_file[image_name]['keypoints'].__array__()
        keypoints += 0.5
        db.add_keypoints(image_id, keypoints)
    
    feature_file.close()
    db.commit()
    db.close()


def in_box(keypoints, box):
    """ Check if keypoint in 2d box, return in_box keypoints indexs"""
    in_box_valid = np.zeros(keypoints.shape[0])

    x0, y0 = box.min(0)
    x1, y1 = box.max(0)
    # x0, y0, x1, y1 = box
    corner0 = np.array([x0, y0])
    corner1 = np.array([x0, y1])
    corner2 = np.array([x1, y0])
    
    vec_01 = corner1 - corner0 
    vec_02 = corner2 - corner0
    vecs = keypoints - corner0

    m1 = np.dot(vecs, vec_01)
    m2 = np.dot(vecs, vec_02)
    for i, (m1_, m2_) in enumerate(zip(m1, m2)):
        if m1_ > 0 and m2_ > 0 and m1_ < np.dot(vec_01, vec_01) and m2_ < np.dot(vec_02, vec_02):
            # ret_idxs.append(i)
            in_box_valid[i] = 1
    return in_box_valid == 1


def import_matches(image_ids, database_path, pairs_path, matches_path, feature_path, match_model='superpoint', \
                   min_match_score=None, skip_geometric_verification=False, verbose=True):
    """ Import matches info into COLMAP database. """
    logging.info("Importing matches into the database...")

    feature_file = h5py.File(str(feature_path), 'r')
    with open(str(pairs_path), 'r') as f:
        pairs = [p.split(' ') for p in f.read().split('\n')]
    
    match_file = h5py.File(str(matches_path), 'r')
    db = COLMAPDatabase.connect(database_path)
    
    if verbose:
        iter_obj = tqdm.tqdm(pairs)
    else:
        iter_obj = pairs
    matched = set()
    for name0, name1 in iter_obj:
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
    
        pair = names_to_pair(name0, name1)
        if pair not in match_file:
            raise ValueError(
                f'Could not find pair {(name0, name1)}... '
                'Maybe you matched with a different list of pairs? '
                f'Reverse in file: {names_to_pair(name0, name1) in match_file}.'
            )
        
        if match_model != 'loftr':
            matches = match_file[pair]['matches0'].__array__()
            valid = matches > -1
        else:
            matches = match_file[pair]['matches'].__array__()
            valid = np.ones((matches.shape[0],), dtype=np.bool) # all True

        if min_match_score:
            scores = match_file[pair]['matching_scores0'].__array__()
            valid = valid & (scores > min_match_score)

        img_type = name0.split('/')[-2]

        if match_model != 'loftr':
            matches = np.stack([np.where(valid)[0], matches[valid]], -1)
        else:
            matches = matches[valid]

        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)
    
    match_file.close()
    db.commit()
    db.close()


def run_triangulation(colmap_path, model_path, database_path, image_dir, empty_model, verbose=True):
    """ run triangulation on given database """
    logging.info('Running the triangulation...')
    
    cmd = [
        str(colmap_path), 'point_triangulator',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--input_path', str(empty_model),
        '--output_path', str(model_path),
        '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_principal_point', '0',
        '--Mapper.ba_refine_extra_params', '0'
    ]
    if verbose:
        logging.info(' '.join(cmd))
        ret = subprocess.call(cmd)
    else:
        ret_all = subprocess.run(cmd, capture_output=True)
        with open(osp.join(model_path, 'output.txt'), 'w') as f:
            f.write(ret_all.stdout.decode())
        ret = ret_all.returncode

    if ret != 0:
        logging.warning('Problem with point_triangulator, existing.')
        exit(ret)
    
    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer', '--path', model_path]
    )
    stats_raw = stats_raw.decode().split('\n')
    stats = dict()
    for stat in stats_raw:
        if stat.startswith('Register images'):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith('Points'):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith('Observation'):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith('Mean track length'):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith('Mean observation per image'):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith('Mean reprojection error'):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])
    return stats


def main(sfm_dir, empty_sfm_model, outputs_dir, pairs, features, matches, match_model='superpoint', \
         colmap_path='colmap', skip_geometric_verification=False, min_match_score=None, image_dir=None, verbose=True):
    """ 
        Import keypoints, matches.
        Given keypoints and matches, reconstruct sparse model from given camera pose.
    """
    assert Path(empty_sfm_model).exists(), empty_sfm_model
    assert Path(features).exists(), features
    assert Path(pairs).exists(), pairs
    assert Path(matches).exists(), matches 

    Path(sfm_dir).mkdir(parents=True, exist_ok=True)
    database = osp.join(sfm_dir, 'database.db')
    model = osp.join(sfm_dir, 'model')
    Path(model).mkdir(exist_ok=True)

    image_ids = create_db_from_model(Path(empty_sfm_model), Path(database))
    import_features(image_ids, database, features, verbose=verbose)
    import_matches(image_ids, database, pairs, matches, features, match_model,
                min_match_score, skip_geometric_verification, verbose=verbose)
    
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs, verbose=verbose)
    
    if not image_dir:
        image_dir = '/'
    stats = run_triangulation(colmap_path, model, database, image_dir, empty_sfm_model, verbose=verbose)
    os.system(f'colmap model_converter --input_path {model} --output_path {outputs_dir}/model.ply --output_type PLY')

@ray.remote(num_cpus=2, num_gpus=1, max_calls=1)  # release gpu after finishing
def main_ray_wrapper(*args, **kwargs):
    return main(*args, **kwargs)