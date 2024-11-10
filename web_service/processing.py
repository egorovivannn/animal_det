from ultralytics import YOLOWorld
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

import argparse
import pandas as pd
import pandas as pd
import glob
import albumentations as A
import cv2
import torch
import torch.nn as nn
import timm
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm





class ModelTrain(nn.Module):
    def __init__(self, vit_backbone, image_size):
        super(ModelTrain, self).__init__()
        self.vit_backbone = vit_backbone
        emb_dim = self.infer_feat_dim(image_size)
        self.linear = nn.Linear(emb_dim, 1)

    def infer_feat_dim(self, image_size):
        return self.vit_backbone(torch.ones([1, 3, image_size, image_size])).shape[-1]
        
    def forward(self, images):
        image_feats = self.vit_backbone(images)
        logits = self.linear(image_feats)
        return logits, image_feats
    

def get_model(model_name, path_to_check, image_size):
    vit_backbone = timm.create_model(model_name, pretrained=True, img_size=(image_size, image_size), num_classes=0)
    model_val = ModelTrain(vit_backbone, image_size)
    model_val.load_state_dict(torch.load(path_to_check))
    model_val.to('cuda')
    model_val.eval()
    return model_val



if __name__ == "__main__":
    # Get path from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="video.mp4")
    args = parser.parse_args()

    path_to_folder = Path(args.folder_path)

    model = YOLOWorld("yolov8x-worldv2.pt")
    model.info()
    model.load('./best (2).pt')
    # model.set_classes(['animal', ])

    model_val_1 = get_model(
        'vit_small_r26_s32_384.augreg_in21k_ft_in1k',
        './all_models_4/best_model#vit_small_r26_s32_384.augreg_in21k_ft_in1k#split_dbscan#0.940279573564929.pth',
        image_size=384
    )

    model_val_2 = get_model(
        'vit_giant_patch14_dinov2.lvd142m',
        './all_models_2/best_model#512#vit_giant_patch14_dinov2.lvd142m#split_dbscan#0.9675903577648102.pth',
        image_size=512
    )
     