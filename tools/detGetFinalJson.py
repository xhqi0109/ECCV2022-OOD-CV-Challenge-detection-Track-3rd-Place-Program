#!/usr/bin/python
# -*- coding: utf-8 -*-
#8719
import json
import os
import shutil
import cv2
import numpy as np
import sys
#/home/data/qxh/dataset/dataset_ECCV2022_OOD/final/final/phase2-det/images
imgDirsPath = str(sys.argv[1]) #'/home/data/lkd/final/phase2-cls/images'
jsonNewPathTrain= str(sys.argv[2])#'/home/data/qxh/dataset/dataset_ECCV2022_OOD/final/phase2-cls.json'
dirs = os.listdir(imgDirsPath)
count = 0
allDict = dict()
ls_images = list()
for dir in dirs:
    count += 1
    imgPath = imgDirsPath+'/'+dir
    print(imgPath)
    img = cv2.imread(imgPath)
    imgDict = dict()
    imgDict["file_name"] = str(dir)
    imgDict["height"] = str(img.shape[0])
    imgDict["width"] = str(img.shape[1])
    imgDict["id"] = str(dir[:-4])
    ls_images.append(imgDict)
print(count)
allDict['images'] = ls_images
allDict['annotations'] = [{"area": 197060, "iscrowd": 0, "bbox": [568, 390, 590, 334], "category_id": 9, "ignore": 0, "segmentation": [], "image_id": "0", "id": 1}]
allDict['type'] = "instances"
allDict['categories'] = [{"supercategory": "none", "id": 1, "name": "aeroplane"}, {"supercategory": "none", "id": 2, "name": "bicycle"}, {"supercategory": "none", "id": 3, "name": "boat"}, {"supercategory": "none", "id": 4, "name": "bus"}, {"supercategory": "none", "id": 5, "name": "car"}, {"supercategory": "none", "id": 6, "name": "chair"}, {"supercategory": "none", "id": 7, "name": "diningtable"}, {"supercategory": "none", "id": 8, "name": "motorbike"}, {"supercategory": "none", "id": 9, "name": "sofa"}, {"supercategory": "none", "id": 10, "name": "train"}]
with open(jsonNewPathTrain,"w") as f:
    json.dump(allDict,f)
    print("加载入文件完成...")
