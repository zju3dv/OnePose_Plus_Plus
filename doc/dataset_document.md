# Dataset Document
## Introduction
The proposed OnePose_LowTexture dataset contains 40 objects with 80 sequences, and eight objects are coupled with a scanned model. The main part of the dataset is placed [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/12121064_zju_edu_cn/EUNsHyFIC7VDhXAYKYokkAIBpqosApirfpVoa7FBs2ogoA?e=Fko6uI), and scanned models are placed [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/12121064_zju_edu_cn/EaLHdKJF45xOu4Kls5eGkqwB3MVd1Pjo0QjLsfh79XIiGw?e=VXYOaA). In our experiment, we use all of the objects for the test.
Note that scanned models are only used for evaluation and to facilitate further research. Our method does not require a known object model.

## Data Structure
The data structure of OnePose_LowTexture is the same as the OnePose dataset:
```
|--- lowtexture_test_data
|       |--- id-objname-category
|               |--- box3d_corners.txt
|               |--- objname-1
|                       |--- Frames.m4v
|                       |--- intrinsics.txt
|                       |--- color
|                       |--- intrin_ba
|                       |--- poses_ba
|                       |--- reproj_box
|               |--- objname-2

```
There are multiple sequences for each object. We use the first sequence (objname-1) for reconstruction and the last sequence (objname-2) for evaluation, similar to OnePose.
For each object:
* `Frames.m4v` is the captured object video.
* `intrinsics.txt` contains an intrinsic matrix of original images in the video. All of the original images share the same intrinsic.
* `box3d_corners.txt` saves eight corners' coordinates of annotated object 3D bounding box.
* `color` directory contains all of the cropped foreground images (resized to $512\times512$). For each cropped image `i.png`, its corresponding intrinsic file and pose file are located in `intrin_ba/i.txt` and `poses_ba/i.txt`, respectively.
* The intrinsic file in the dataset contains the $3\times3$ projection matrix of the corresponding image. And the pose is defined as a $4\times4$ homogeneous transformation from the object system to the camera coordinate system.
