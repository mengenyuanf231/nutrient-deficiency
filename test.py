import sys
import ultralytics
import os

# 查看 ultralytics 包结构
#print(dir(ultralytics))
from ultralytics import YOLO

#os.environ['CUDA_VISIBLE_DEVICES']=2,3

#print(dir(ultralytics))

import ultralytics.models

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Build a YOLOv8n model from scratch

model = YOLO("/root/autodl-tmp/code/ultralytics_sunshi/epoch194.pt")
model.info()  # Display model information
model.val(
              data='ultralytics/cfg/datasets/aug.yaml',
              split='test',
              imgsz=640,
              batch=8,              
              device='0', ##
              name='runs/test/back-neckhs/epoch128',
       
       )