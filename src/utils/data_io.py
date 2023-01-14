import pickle
import h5py
import numpy as np
import os
import os.path as osp
import cv2
import torch

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_h5(file_path, transform_slash=True, parallel=False):
    """load the whole h5 file into memory (not memmaped)
    """
    with h5py.File(file_path, 'r') as f:
        # if parallel:
        #     Parallel()
        data = {k if not transform_slash else k.replace('+', '/'): v.__array__() \
                    for k, v in f.items()}
    return data

def save_h5(dict_to_save, filename, transform_slash=True):
    """Saves dictionary to hdf5 file"""
    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:  # h5py doesn't allow '/' in object name (will leads to sub-group)
            f.create_dataset(key.replace('/', '+') if transform_slash else key,
                             data=dict_to_save[key])

def read_grayscale(path, resize=None, resize_float=False, df=None,
                   pad_to=None, ret_scales=False, ret_pad_mask=False,
                   augmentor=None):
    resize = tuple(resize) if resize is not None else None
    assert osp.exists(path), f"image path: {path} not exists!"
    if augmentor is None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augmentor(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    assert image is not None, f"path: {path} image not properly loaded"
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize, df)
    scales = torch.tensor([float(h) / float(h_new), float(w) / float(w_new)]) # [2]

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if pad_to is not None:
        image, mask = pad_bottom_right(image, pad_to, ret_mask=ret_pad_mask)

    ts_image = grayscale2tensor(image)
    ret_val = [ts_image]

    if ret_scales:
        ret_val.append(scales)
    if ret_pad_mask:
        ts_mask = mask2tensor(mask) if pad_to else None
        ret_val.append(ts_mask if pad_to else None)
    return ret_val[0] if len(ret_val) == 1 else ret_val

def process_resize(w, h, resize, df=None):
    if resize is not None:
        assert(len(resize) > 0 and len(resize) <= 2)
        if len(resize) == 1 and resize[0] > -1:  # resize the larger side
            scale = resize[0] / max(h, w)
            w_new, h_new = int(round(w*scale)), int(round(h*scale))
        elif len(resize) == 1 and resize[0] == -1:
            w_new, h_new = w, h
        else:  # len(resize) == 2:
            w_new, h_new = resize[0], resize[1]
    else:
        w_new, h_new = w, h

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])
    return w_new, h_new

def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:])
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=inp.dtype)
            mask[:inp.shape[0], :inp.shape[1]] = 1
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
            mask[:, :inp.shape[1], :inp.shape[2]] = 1
    else:
        raise NotImplementedError()
    return padded, mask

def grayscale2tensor(image, mask=None):
    return torch.from_numpy(image/255.).float()[None]  # (1, h, w)
    
def mask2tensor(mask):
    return torch.from_numpy(mask).float()  # (h, w)