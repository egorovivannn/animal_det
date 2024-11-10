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

image_size = 512
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
augs = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=mean, std=std, p=1)
])


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
    
    # model_val_2 = get_model(
    #     'vit_small_r26_s32_384.augreg_in21k_ft_in1k',
    #     './all_models_4/best_model#vit_small_r26_s32_384.augreg_in21k_ft_in1k#split_dbscan#0.940279573564929.pth',
    #     image_size=384
    # )

    table = []
    for file_name in tqdm(list(path_to_folder.iterdir())):
        img = plt.imread(file_name)
        r = model(img, verbose=False, half=True)[0]

        crops = []
        bboxes_normalized = r.boxes.xywhn
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
            crop = img[y1:y2, x1:x2]
        
            crop = augs(image=crop)['image']
            crop = torch.from_numpy(crop).permute(2, 0, 1)

            crops.append(crop)

        if len(crops) == 0:
            continue

        images = torch.stack(crops)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            images = images.to('cuda')
            outputs1 = model_val_1(torch.nn.functional.interpolate(images, size=(384, 384)))[0] 
            outputs2 = model_val_2(images)[0] 
            outputs = 0.5 * outputs1 + 0.5 * outputs2
            outputs_bin = (outputs.sigmoid() > 0.5).flatten().cpu().detach().numpy().astype('int').tolist()
        
        for ind in range(len(bboxes_normalized)):
            bbox_str = ",".join(map(str, bboxes_normalized[ind].tolist()))
            table.append([file_name.name, bbox_str, outputs_bin[ind]])

    df = pd.DataFrame(table, columns=['Name', 'Bbox', 'Class'])
    df.to_csv('table_test.csv', index=False)
