#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import shutil
import cv2
import numpy as np
import imageio
import sys

imgDirsPath = str(sys.argv[1]) #'/home/data/qxh/dataset/dataset_ECCV2022_OOD/final/final/phase2-det/images'

dirs = os.listdir(imgDirsPath)

for dir in dirs:
    imgPath = imgDirsPath+'/'+dir
    img = cv2.imread(imgPath)
    #图像为空
    if np.max(img) == None:
        print(imgPath)            
        gif = imageio.mimread(imgPath)
        nums = len(gif)
        print("Total {} frames in the gif!".format(nums))
        # convert form RGB to BGR
        imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
        #重新写替换,取动图的第0帧，来替换gif
        cv2.imwrite(imgPath,imgs[0])
print("check over")