from loguru import logger
from src.datasets.OnePosePlus_dataset import OnePosePlusDataset
from src.utils.utils_phoaug import build_augmentor
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
import os.path as osp


class OnePosePlusDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.train_anno_file = kwargs["train_anno_file"]
        self.val_anno_file = kwargs["val_anno_file"]
        assert osp.exists(self.train_anno_file), self.train_anno_file
        if not osp.exists(self.val_anno_file):
            logger.warning(
                f"Val anno path: {self.val_anno_file} not exists! use train anno instead"
            )
            self.val_anno_file = self.train_anno_file

        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.pin_memory = kwargs["pin_memory"]
        self.augmentor_method = kwargs['augmentor_method']

        # Data related
        self.train_percent = kwargs["train_percent"]
        self.val_percent = kwargs["val_percent"]
        self.train_image_warp_adapt = kwargs['train_image_warp_adapt']
        # 3D part
        self.shape3d_train = kwargs["shape3d_train"]
        self.shape3d_val = kwargs["shape3d_val"]
        self.load_3d_coarse = kwargs["load_3d_coarse"]
        # 2D part
        self.img_pad = kwargs["img_pad"]
        self.img_resize = kwargs["img_resize"]
        self.df = kwargs["df"]
        self.coarse_scale = kwargs["coarse_scale"]

        # Loader parameters:
        self.train_loader_params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.val_loader_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.test_loader_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.augmentor = build_augmentor(self.augmentor_method)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """ Load data. Set variable: self.data_train, self.data_val, self.data_test"""
        train_set = OnePosePlusDataset(
            anno_file=self.train_anno_file,
            img_pad=self.img_pad,
            img_resize=self.img_resize,
            coarse_scale=self.coarse_scale,
            df=self.df,
            shape3d=self.shape3d_train,
            percent=self.train_percent,
            split='train',
            load_pose_gt=True,
            load_3d_coarse_feature=self.load_3d_coarse,
            image_warp_adapt=self.train_image_warp_adapt,
            augmentor=self.augmentor
        )
        print("=> Read train anno file: ", self.train_anno_file)

        val_set = OnePosePlusDataset(
            anno_file=self.val_anno_file,
            pad=True,
            img_pad=self.img_pad,
            img_resize=self.img_resize,
            coarse_scale=self.coarse_scale,
            df=self.df,
            shape3d=self.shape3d_val,
            percent=self.val_percent,
            split='val',
            load_pose_gt=True,
            load_3d_coarse_feature=self.load_3d_coarse
        )

        self.data_train = train_set
        self.data_val = val_set
        self.data_test = val_set

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.train_loader_params)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, **self.val_loader_params)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, **self.test_loader_params)
