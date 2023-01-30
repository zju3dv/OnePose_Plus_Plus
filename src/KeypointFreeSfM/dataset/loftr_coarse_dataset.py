import os
import random
import os.path as osp
import torch

from torch.utils.data import Dataset
from src.utils.data_io import (
    read_grayscale,
)


class LoftrCoarseDataset(Dataset):
    """Build image Matching Challenge Dataset image pair (val & test)"""

    def __init__(
        self,
        args,
        image_lists,
        covis_pairs,
    ):
        """
        Parameters:
        ---------------
        """
        super().__init__()
        self.img_dir = image_lists
        self.img_resize = args['img_resize']
        self.df = args['df']
        shuffle = args['shuffle']

        if isinstance(covis_pairs, list):
            self.pair_list = covis_pairs
        else:
            assert osp.exists(covis_pairs)
            # Load pairs: 
            with open(covis_pairs, 'r') as f:
                self.pair_list = f.read().rstrip('\n').split('\n')

        if shuffle:
            random.shuffle(self.pair_list)

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        img_path0, img_path1 = self.pair_list[idx].split(' ')
        img_scale0 = read_grayscale(
            img_path0,
            (self.img_resize,) if self.img_resize is not None else None,
            df=self.df,
            ret_scales=True,
        )
        img0, scale0 = map(lambda x: x[None], img_scale0)  # no dataloader operation
        img_scale1 = read_grayscale(
            img_path1,
            (self.img_resize,) if self.img_resize is not None else None,
            df=self.df,
            ret_scales=True,
        )
        img1, scale1 = map(lambda x: x[None], img_scale1)  # no dataloader operation

        data = {
            "image0": img0,  # 1*1*H*W because no dataloader operation, if batch: 1*H*W
            "image1": img1,
            "scale0": scale0,  # 1*2
            "scale1": scale1,
            "f_name0": osp.basename(img_path0).rsplit('.', 1)[0],
            "f_name1": osp.basename(img_path1).rsplit('.', 1)[0],
            "frameID": idx,
            "pair_key": (img_path0, img_path1),
        }

        return data

