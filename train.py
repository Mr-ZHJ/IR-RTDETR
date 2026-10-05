# -*- coding: utf-8 -*-
import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    # 加载改进模型配置
    model = RTDETR('ultralytics/cfg/models/hw-detr/FMA1.yaml')
    # model.load('') # 加载预训练权重
    model.train(data='dataset/uav410sub01.yaml',  # Anti-UAV410 红外无人机数据集
        cache=False,
        imgsz=640,       # 输入分辨率 640×640
        rect=True,       # 矩形矫正
        epochs=100,      # 训练 100 轮
        batch=4,         # 批次大小
        workers=4,       # 数据加载线程
        # patience=30,   # 早停
        # warmup_epochs=10,  # 预热轮数
        # cos_lr=True,   # 余弦退火学习率
        # resume='',     # 断点续训 last.pt 路径
        project='runs/train/20260228',
        name='MANet_FasterBlock-n1',
        )