import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    model = YOLO(r'/root/autodl-tmp/code/ultralytics_sunshi/ultralytics/cfg/models/mymodel/he/back-head.yaml')
    model.train(data=r'ultralytics/cfg/datasets/aug.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                single_cls=False,  # 是否是单类别检测 /root/autodl-tmp/quesu/ultralytics_sunshi/train.py               
                batch=8,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',
                amp=False,
                resume=False,
              
                name='/root/autodl-tmp/code/ultralytics_sunshi/runs/back-head',

                )