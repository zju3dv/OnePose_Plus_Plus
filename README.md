# OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models
### [Project Page](https://zju3dv.github.io/onepose_plus_plus) | [Paper](https://openreview.net/pdf?id=BZ92dxDS3tO)
<br/>

> OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models                                                                               
> [Xingyi He](https://github.com/hxy-123/)<sup>\*</sup>, [Jiaming Sun](https://jiamingsun.ml)<sup>\*</sup>, [Yu'ang Wang](https://github.com/angshine), [Di Huang](https://github.com/dihuangdh), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/), [Xiaowei Zhou](https://xzhou.me)                              
> NeurIPS 2022

![demo_vid](assets/demo.gif)

## TODO List
- [x] Training, inference and demo code.
- [x] Pipeline to reproduce the evaluation results on the OnePose dataset and proposed OnePose_LowTexture dataset.
- [x] `OnePose Cap` app is available at the [App Store](https://apps.apple.com/cn/app/onepose-capture/id6447052065?l=en-GB) (iOS only) to capture your own training and test data.

## Installation

```shell
conda env create -f environment.yaml
conda activate oneposeplus
```

[LoFTR](https://github.com/zju3dv/LoFTR) and [DeepLM](https://github.com/hjwdzh/DeepLM) are used in this project. Thanks for their great work, and we appreciate their contribution to the community. Please follow their installation instructions and LICENSE:
```shell
git submodule update --init --recursive

# Install DeepLM
cd submodules/DeepLM
sh example.sh
cp ${REPO_ROOT}/backup/deeplm_init_backup.py ${REPO_ROOT}/submodules/DeepLM/__init__.py
```
Note that the efficient optimizer DeepLM is used in our SfM refinement phase. If you face difficulty in installation, do not worry. You can still run the code by using our first-order optimizer, which is a little slower.

[COLMAP](https://colmap.github.io/) is also used in this project for Structure-from-Motion. Please refer to the official [instructions](https://colmap.github.io/install.html) for the installation.

Download the [pretrained models](https://drive.google.com/drive/folders/1tV-w9Wpz0FKQsW8-3RhqcG-_EI9dO171?usp=sharing), including our 2D-3D matching and LoFTR models. Then move them to `${REPO_ROOT}/weights`.

[Optional] You may optionally try out our web-based 3D visualization tool [Wis3D](https://github.com/zju3dv/Wis3D) for convenient and interactive visualizations of feature matches and point clouds. We also provide many other cool visualization features in Wis3D, welcome to try it out.

```bash
# Working in progress, should be ready very soon, only available on test-pypi now.
pip install -i https://test.pypi.org/simple/ wis3d
```
## Demo
After the installation, you can refer to [this page](doc/demo.md) to run the demo with your custom data.


## Training and Evaluation
### Dataset setup 
1. Download OnePose dataset from [here](https://drive.google.com/drive/folders/1D11oh8BXOfbsbGCtJ46z8EzZBQca-Uce) and OnePose_LowTexture dataset from [here](https://drive.google.com/file/d/12CTxpKKskhbw40eR15tlIzl54DGeVwMi/view?usp=sharing), and extract them into `$/your/path/to/onepose_datasets`. 
If you want to evaluate on LINEMOD dataset, download the real training data, test data and 3D object models from [CDPN](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi), and detection results by YOLOv5 from [here](https://drive.google.com/file/d/1s-OQv6mWgEvRHjPGABH0e29Xb8SLhQHW/view?usp=sharing). Then extract them into `$/your/path/to/onepose_datasets/LINEMOD`
The directory should be organized in the following structure:
    ```
    |--- /your/path/to/datasets
    |       |--- train_data
    |       |--- val_data
    |       |--- test_data
    |       |--- lowtexture_test_data
    |       |--- LINEMOD
    |       |      |--- real_train
    |       |      |--- real_test
    |       |      |--- models
    |       |      |--- yolo_detection
    ```
You can refer to [dataset document](doc/dataset_document.md) for more informations about OnePose_LowTexture dataset.

2. Build the dataset symlinks
    ```shell
    REPO_ROOT=/path/to/OnePose_Plus_Plus
    ln -s /your/path/to/datasets $REPO_ROOT/data/datasets
    ```
### Reconstruction
Reconstructed the semi-dense object point cloud and 2D-3D correspondences are needed for both training and test objects:
```python
python run.py +preprocess=sfm_train_data.yaml use_local_ray=True  # for train data
python run.py +preprocess=sfm_inference_onepose_val.yaml use_local_ray=True # for val data
python run.py +preprocess=sfm_inference_onepose.yaml use_local_ray=True # for test data
python run.py +preprocess=sfm_inference_lowtexture.yaml use_local_ray=True # for lowtexture test data
```
### Inference
```shell
# Eval OnePose dataset:
python inference.py +experiment=inference_onepose.yaml use_local_ray=True verbose=True

# Eval OnePose_LowTexture dataset:
python inference.py +experiment=inference_onepose_lowtexture.yaml use_local_ray=True verbose=True
```
Note that we perform the parallel evaluation on a single GPU with two workers by default. If your GPU memory is smaller than 6GB, you are supposed to add `use_local_ray=False` to turn off the parallelization.

### Evaluation on LINEMOD Dataset
```shell
# Parse LINDMOD Dataset to OnePose Dataset format:
sh scripts/parse_linemod_objs.sh

# Reconstruct SfM model on real training data:
python run.py +preprocess=sfm_inference_LINEMOD.yaml use_local_ray=True

# Eval LINEMOD dataset:
python inference.py +experiment=inference_LINEMOD.yaml use_local_ray=True verbose=True
```

### Training
1. Prepare ground-truth annotations. Merge annotations of training/val data:
    ```python
    python merge.py +preprocess=merge_annotation_train.yaml
    python merge.py +preprocess=merge_annotation_val.yaml
    ```
   
2. Begin training
    ```python
    python train_onepose_plus.py +experiment=train.yaml exp_name=onepose_plus_train
    ```
    Note that the default config for training uses 8 GPUs with around 23GB VRAM for each GPU. You can set the GPU number or ID in `trainer.gpus` and reduce the batch size in `datamodule.batch_size` to reduce the GPU VRAM footprint.
   
All model weights will be saved under `${REPO_ROOT}/models/checkpoints/${exp_name}` and logs will be saved under `${REPO_ROOT}/logs/${exp_name}`.
You can visualize the training process by Tensorboard:
```shell
tensorboard --logdir logs --bind_all --port your_port_number
```

## Citation
If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{
    he2022oneposeplusplus,
    title={OnePose++: Keypoint-Free One-Shot Object Pose Estimation without {CAD} Models},
    author={Xingyi He and Jiaming Sun and Yuang Wang and Di Huang and Hujun Bao and Xiaowei Zhou},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```


## Acknowledgement
Part of our code is borrowed from [hloc](https://github.com/cvg/Hierarchical-Localization) and [LoFTR](https://github.com/zju3dv/LoFTR). Thanks to their authors for their great works.
