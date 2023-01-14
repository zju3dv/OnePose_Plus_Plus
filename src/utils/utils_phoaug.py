import cv2
from pathlib import Path
from itertools import chain
from functools import partial
from loguru import logger

import albumentations as A
import numpy as np


class darkAug(object):
    """
    Note: original lambda function cannot be pickled in ddp_spwan
    """

    def __init__(self) -> None:
        self.augmentor = A.Compose(
            [
                A.RandomBrightnessContrast(
                    p=0.75, brightness_limit=(-0.6, 0.0), contrast_limit=(-0.5, 0.3)
                ),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.MotionBlur(p=0.2, blur_limit=(3, 25)),
                A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
                A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40)),
            ],
            p=0.75,
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


class MobileAug(object):
    """
    Random augmentations aiming at images of mobile/handhold devices.
    """

    def __init__(self):
        self.augmentor = A.Compose(
            [
                A.MotionBlur(p=0.25),
                A.ColorJitter(p=0.5),
                A.RandomRain(p=0.1),  # random occlusion
                # A.RandomSunFlare(p=0.1),
                A.JpegCompression(p=0.25),
                A.ISONoise(p=0.25),
            ],
            p=1.0,
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


class YCBAug(object):
    """
    """

    def __init__(self):
        self.augmentor = A.Compose(
            [
                A.ISONoise(intensity=(0.4, 0.9),p=0.25),
                A.GaussNoise(var_limit=(100, 300),p=0.7),
                A.GaussianBlur(sigma_limit=10 ,p=0.7),
            ],
            p=1.0,
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


class Stylization(object):
    def __init__(self, ref_root="assets/isrf", method="FDA", beta_limit=0.05, p=0.5):
        self.method = method
        logger.info(f"Loading reference images...")
        f_names = list(
            chain(
                *[
                    Path(ref_root).glob(f"**/*.{ext}")
                    for ext in ["png", "jpg", "jpeg", "JPEG"]
                ]
            )
        )
        self.ref_imgs = [
            cv2.cvtColor(cv2.imread(str(ref_root / fn)), cv2.COLOR_BGR2RGB)
            for fn in f_names
        ]
        logger.info(
            f"Using {method} stylization with {len(self.ref_imgs)} reference images."
        )
        self.stylizer = self.build_stylizer(method, beta_limit, p)

    @staticmethod
    def build_stylizer(method, beta_limit=0.05, p=0.5):
        if method == "FDA":
            return partial(A.FDA, beta_limit=beta_limit, p=p, read_fn=lambda x: x)
        else:
            raise NotImplementedError()

    def __call__(self, x):
        ref_img = np.random.choice(self.ref_imgs, replace=True)
        aug = A.Compose([self.stylizer([ref_img])])
        return aug(image=x)["image"]


def build_augmentor(method=None, **kwargs):
    if method == "dark":
        return darkAug()
    elif method == "mobile":
        return MobileAug()
    elif method == "FDA":  # fourier domain adaptation
        return Stylization(method="FDA", **kwargs)
    elif method == "Gaussian":
        return GaussianAug()
    elif method is None:
        return None
    else:
        raise ValueError(f"Invalid augmentation method: {method}")


if __name__ == "__main__":
    augmentor = build_augmentor("FDA")

