import argparse
import yaml
import pandas as pd
import albumentations as A
import utilities
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import timm
import math
from transformers import (get_cosine_schedule_with_warmup)
from tqdm import tqdm
import random
import gc
import os
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np


PATH2DATASET = 'train_data_minprirodi/images/'
DF_PATH = './annotation_split_clusters.csv'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert_bbox(xc, yc, w, h, img_width, img_height):
    x1 = int((xc - w / 2) * img_width)
    y1 = int((yc - h / 2) * img_height)
    x2 = int((xc + w / 2) * img_width)
    y2 = int((yc + h / 2) * img_height)
    return x1, y1, x2, y2


class MetalDataset(Dataset):
    def __init__(self, 
                 df, 
                 mode='train', 
                 split_col = 'split',
                 scale = 1.1,
                 p_neg_class: float = 1.0,
                 cross_val_num = None,                  
                 p_resize=0.5, 
                 transform=None
        ):
        """
        MetalDataset is the main class for creating train and val datasets. 

        :param df: DataFrame with all the data 
        :param mode: dataset mode ('train' или 'val'). 
        :param split_col: worlks in two modes. It can be either a name of column 
        which should contain 'train' or 'val' value for each bounding box in a dataset. 
        Or it can be a cross validation class with number from 0 to N.  
        :param scale: how much to widen the bbox 
        :param p_neg_class: the probability of generating bad examples from the positive clases by augmentation
        :param cross_val_num: cros val index of a val dataset. If mode is train all the 
        data except data with this index is used for training. 
        :param p_resize: the probability that during the process of generating negative class examples
        from positive ones the image is going to be resized. The probability of blur is 1 - p_resize. 
        :param transform: augmentation transform
        """

        if cross_val_num is None:
            self.df = df[df[split_col]==mode].reset_index(drop=True)
        else:
            if mode == 'train':
                self.df = df[df[split_col] != cross_val_num ].reset_index(drop=True)
            else:
                self.df = df[df[split_col] == cross_val_num ].reset_index(drop=True)

        self.scale = scale
        self.augs = transform
        self.mode = mode
        self.cache_images()
        self.paths = list(self.df.fname.unique())
        self.p_neg_class = p_neg_class
        self.p_resize = p_resize

    def cache_images(self):
        self.idx2meta = {}
        idx = 0
        scale = self.scale
        for path in tqdm(self.df.fname.unique()):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bboxes  = df[df['fname']==path][['x1', 'y1', 'x2', 'y2']].values.tolist()
            labels = df[df['fname']==path]['Class'].values.tolist()

            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox
                x1 = int(x1/scale)
                x2 = int(x2*scale)    
                y1 = int(y1/scale)    
                y2 = int(y2*scale)  

                crop = img[y1:y2, x1:x2]
                
                self.idx2meta[idx] = {
                    'img': crop,
                    'label': label,
                    'path': path
                }
                idx+=1

    def __len__(self):
        return len(self.idx2meta)
    
    def __getitem__(self, idx):
        item = self.idx2meta[idx]
        img = item['img']
        label = item['label']
        path = item['path']
        
        try:
            augmented_img = None
            augmented_label = label
            # If the mode is train and the current example is positive we can generate an artifical negative example from it
            if label == 1 and random.random() < self.p_neg_class and self.mode == 'train':
                augmented_img = img.copy()
                if random.random() < self.p_resize:
                    augmented_img = A.RandomResizedCrop(height=img.shape[0], width=img.shape[1], scale=(0.1, 0.5))(image=augmented_img)['image']
                else:
                    augmented_img = A.GaussianBlur(blur_limit=(11, 21), p=1.0)(image=augmented_img)['image']
                augmented_label = 0

            if self.augs:
                img = self.augs(image=img)['image']
                img = torch.from_numpy(img).permute(2, 0, 1)
                if augmented_img is not None:
                    augmented_img = self.augs(image=augmented_img)['image']
                    augmented_img = torch.from_numpy(augmented_img).permute(2, 0, 1)
        except Exception as e:
            print(e, path)
            img = torch.zeros((3, 224, 224))
            if augmented_img is not None:
                augmented_img = torch.zeros((3, 224, 224))
        
        if augmented_img is not None:
            return [(img, label, path), (augmented_img, augmented_label, path)]
        else:
            return [(img, label, path), ]


def get_augs(mean, std, image_size, mode='train'):
    # Only simple augs are used, so not to distort the image quality
    if mode=='train':
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std, p=1), 
            ])

    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std, p=1)
        ])
        

class ModelTrain(nn.Module):
    def __init__(self, vit_backbone, image_size):
        # emb dim is determined using image size 
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


def train(model, train_loader, optimizer, scaler, scheduler, epoch, criterion, cfg_dict, writer, pbar_draw=False):
    model.train()
    loss_metrics = AverageMeter()
    bar = tqdm(train_loader, disable=not pbar_draw)
    for step, data in enumerate(bar):
        step += 1
        with torch.amp.autocast(device_type='cuda', enabled=cfg_dict['autocast']):
            images = data[0].to(cfg_dict['device'])
            labels = data[1].to(cfg_dict['device'])
            batch_size = labels.size(0)
            outputs, features = model(images)

        loss = criterion(outputs, labels[:, None].float())
        loss_metrics.update(loss.item(), batch_size)
        loss = loss / cfg_dict['acc_steps']
        scaler.scale(loss).backward()

        if step % cfg_dict['acc_steps'] == 0 or step == len(bar):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            cfg_dict['global_step'] += 1
            
        lrs = get_lr_groups(optimizer.param_groups)
        loss_avg = loss_metrics.avg
        bar.set_postfix(loss=loss_avg, epoch=epoch, lrs=lrs, step=cfg_dict['global_step'])
        
        writer.add_scalar('Train/Loss', loss_avg, cfg_dict['global_step'])
        writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'], cfg_dict['global_step'])

        
def validate(model, loader):
    model.eval()
    device = 'cuda'
    preds = []
    labels = []
    for i, (images, label, paths) in enumerate(loader):
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            outputs = model(images.to(device))
            outputs = (outputs[0].sigmoid() > 0.5).flatten().cpu().detach().numpy().astype('int').tolist()
            preds.extend(outputs)
            labels.extend(label)
            
    metric = f1_score(labels, preds, average='macro')
    return metric


def get_lr_groups(param_groups):
    groups = sorted(set([param_g['lr'] for param_g in param_groups]))
    groups = ["{:2e}".format(group) for group in groups]
    return groups


def collate_fn(batches):
    # As our dataset returns not a tuple but a list of tuples, 
    # custom collate function is needed. 
    output_lists = [[], [], []]
    for b in batches:
        for subl in b:
            for ind, el in enumerate(subl):
                output_lists[ind].append(el)
    images, labels, paths = output_lists
    return torch.stack(images), torch.tensor(labels), paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration from a file.")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as file:
        cfg_dict = yaml.safe_load(file)
        
    writer = SummaryWriter(log_dir=cfg_dict["log_dir"])
        
    mean = [0.5, 0.5, 0.5]
    image_size = cfg_dict['image_size']
    split_col = cfg_dict["split_col"]
    
    df = pd.read_csv(DF_PATH)
    utilities.set_seed(cfg_dict['seed'])
    vit_backbone = timm.create_model(cfg_dict["model_name"], pretrained=True, img_size=(image_size, image_size), num_classes=0)
    model = ModelTrain(vit_backbone, image_size)
    model.to('cuda')

    train_dataset = MetalDataset(
        df,
        mode='train', 
        split_col=split_col, 
        scale=cfg_dict['scale'],
        cross_val_num=cfg_dict["cross_val_num"],
        p_neg_class=cfg_dict['p_neg_class'],
        p_resize=cfg_dict["p_resize"],
        transform=get_augs(mean, mean, image_size, 'train')
    )

    # We also create a validation version of train dataset to track the metric on it
    train_dataset_for_val = MetalDataset(
        df, 
        mode='train', 
        split_col=split_col,
        scale=cfg_dict['scale'], 
        cross_val_num=cfg_dict["cross_val_num"],
        p_neg_class=-1.0,
        transform=get_augs(mean, mean, image_size, 'val')
    )
    
    valid_dataset = MetalDataset(
        df, 
        mode='val', 
        split_col=split_col,
        scale=cfg_dict['scale'],
        cross_val_num=cfg_dict["cross_val_num"],
        p_neg_class=-1.0,
        transform=get_augs(mean, mean, image_size, 'val')
    )

    train_loader = DataLoader(train_dataset, batch_size = cfg_dict['train_batch_size'], num_workers=cfg_dict['workers'], shuffle=True, drop_last=False, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size = cfg_dict['valid_batch_size'], num_workers=cfg_dict['workers'], shuffle=False, drop_last=False, collate_fn=collate_fn)
    train_valid_loader = DataLoader(train_dataset_for_val, batch_size = cfg_dict['valid_batch_size'], num_workers=cfg_dict['workers'], shuffle=False, drop_last=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_dict["lr"], weight_decay=cfg_dict["wd"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg_dict['autocast'])
    steps_per_epoch = math.ceil(len(train_loader) / cfg_dict['acc_steps'])
    num_training_steps = math.ceil(cfg_dict['n_epochs'] * steps_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_training_steps=num_training_steps,
                                                num_warmup_steps=cfg_dict['n_warmup_steps'])   
    
    criterion = F.binary_cross_entropy_with_logits

    cfg_dict['global_step'] = 0    
    best_score = 0

    epoch = -1
    score = validate(model, valid_loader)
    print(f'Epoch = {epoch}, score:', score)
    for epoch in range(math.ceil(cfg_dict['n_epochs'])):
        train(model, train_loader, optimizer, scaler, scheduler, epoch, criterion, cfg_dict, writer, pbar_draw=True)
        score = validate(model, valid_loader)
        writer.add_scalar('Validation/F1 Score', score, epoch)
        score2 = validate(model, train_valid_loader)
        writer.add_scalar('Validation/Train F1 Score', score2, epoch)
        print(f'Epoch = {epoch}, score:', score, 'score train', score2)
        
        # Here we save the best model by metrics. 
        # To save the space each all models save to the folders which are assigned to the specific gpus
        if score > best_score:
            best_score = score
            path_to_models = f"new_all_models_{os.environ['CUDA_VISIBLE_DEVICES']}" 
            os.makedirs(path_to_models, exist_ok=True)
            found_same = 0
            for path_to_weight in os.listdir(path_to_models):
                if '#' not in path_to_weight:
                    continue
                from pathlib import Path
                try:
                    _, image_size_saved, model_name_saved, split_col_saved, best_score_str = Path(path_to_weight).stem.split("#")
                except:
                    print("skip", path_to_weight)
                    continue
                print(path_to_weight, score, model_name_saved, float(best_score_str), split_col_saved)
                if (cfg_dict["model_name"] == model_name_saved) and (split_col_saved == split_col) and (int(image_size_saved) == image_size):
                    found_same += 1
                if (cfg_dict["model_name"] == model_name_saved) and (split_col_saved == split_col) and (int(image_size_saved) == image_size) and (score > float(best_score_str)):
                    print("UPDATE",  f'best_model#{cfg_dict["model_name"]}#{split_col}#{best_score}.pth')
                    torch.save(model.state_dict(), os.path.join(path_to_models,  f'best_model#{image_size}#{cfg_dict["model_name"]}#{split_col}#{best_score}.pth'))
                    os.remove(os.path.join(path_to_models, path_to_weight))
            if found_same == 0:
                torch.save(model.state_dict(), os.path.join(path_to_models, f'best_model#{image_size}#{cfg_dict["model_name"]}#{split_col}#{best_score}.pth'))
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # here we can stop early, as last epochs of cosine optimizer doesn't change metric too much
        if "real_epoch_num" in cfg_dict and epoch > cfg_dict["real_epoch_num"]:
            break
    
    writer.close()
