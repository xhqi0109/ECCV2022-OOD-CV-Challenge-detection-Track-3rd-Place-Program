# ECCV2022-OOD-CV-Challenge-detection-Track-3rd-Place-Program
Competition Open Source Solutions


## 1. Environment setting 

### 1.0. Package
* Several important packages
    - torch == 1.8.1+cu111
    - trochvision == 0.9.1+cu111

### 1.1. Dataset
In the classification track, we use only the OOD detection data and labels:
* [ECCV-OOD](https://github.com/eccv22-ood-workshop/ROBIN-dataset)

### 1.2. OS
- [x] Windows10
- [x] Ubuntu20.04
- [x] macOS (CPU only)

## 2. Train
- [ ] Single GPU Training
- [x] DataParallel (single machine multi-gpus)
- [ ] DistributedDataParallel

(more information: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### 2.1. data
train data and test data structure:  
```
├── data/
│   ├── train
|   |    ├── Images
│   |    └── train.json
│   ├── test
|   |    ├── Images
│   |    └── phase2-det.json
│   ├── val
|   |    ├── Images
│   |    └── iid_test.json
│   └── occlusion
```
The structure flow of the generated file is as follows：
#### 2.1.1 move picture
Put the training set image "ROBINv1.1-det" in
```
./data/train/Images/ 
```
Put the test set picture "phase2-det" in
```
./data/test/Images/
```
#### 2.1.2 check picture
Since the given dataset contains gif images, we need to convert gif images into jpg images. We take the first frame of gif images as jpg images, and the generated jpg images automatically replace the original gif images.
```
python ./tools/check_gif.py  ./data/test/Images
```
### 2.2. run.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh  \
./config/cascade_rcnn_r50_fpn_1x_coco_backbone_convnextLarge_OnlyAdamW_cos_colorjitter_softmax_corrupt.py 8 \
--seed 0 \ 
--deterministic \ 
--work-dir ./work_dirs/  
```

## 3. Evaluation

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test_final.sh \
./config/cascade_rcnn_r50_fpn_1x_coco_backbone_convnextLarge_OnlyAdamW_cos_colorjitter_softmax_corrupt.py \ 
./work_dirs/epoch_15.pth 8 \
--format-only \ 
--options "jsonfile_prefix=./work_dirs/out" > ./work_dirs/out-test.out & 
```

## 4. Challenge's final checkpoints
It can be downloaded from Baidu Cloud Disk: https://pan.baidu.com/s/1scW9Z-PjZbrqi3VL7SNJ5w 
Extraction code：tc77

It can be directly used for model reasoning and get final result.

### Acknowledgment

* Thanks to [timm](https://github.com/rwightman/pytorch-image-models) for Pytorch implementation.
