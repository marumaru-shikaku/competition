#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_rows = 200
pd.options.display.max_columns = 100
import numpy as np
np.set_printoptions(threshold=100)
import cv2
from PIL import Image
import os
import glob
import random
import gc
import optuna
import timm
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations import BasicTransform
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

def score(solution: np.ndarray, submission: np.ndarray, min_tpr: float=0.80) -> float:
    v_gt = abs(solution-1)
    v_pred = np.array([1.0 - x for x in submission])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

def cutmix(x, y, area_percentage: float = 0.50):
    vertical_center = x.shape[2] // 2
    horizontal_center = x.shape[3] // 2
    
    axis_percentage = np.sqrt(area_percentage)
    vertical_limit = int(axis_percentage/2 * x.shape[2])
    horizontal_limit = int(axis_percentage/2 * x.shape[3])
    
    upper_limit = np.random.randint(0, vertical_center-vertical_limit)
    lower_limit= np.random.randint(vertical_center+vertical_limit, x.shape[2])
    left_limit = np.random.randint(0, horizontal_center-horizontal_limit)
    right_limit= np.random.randint(horizontal_center+horizontal_limit, x.shape[3])
    
    indices = torch.randperm(x.shape[0])
    new_x = x.clone()
    new_y = y[indices]
    new_x[:, :, upper_limit:lower_limit, left_limit:right_limit] = x[indices, :, upper_limit:lower_limit, left_limit:right_limit]
    
    lam = 1 - ((lower_limit-upper_limit)*(right_limit-left_limit)) / (x.shape[2]*x.shape[3])

    return new_x, y, new_y, lam

def mix_criterion(y_pred, y, new_y, lam, criterion):
    return lam * criterion(y_pred, y) + (1 - lam) * criterion(y_pred, new_y)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)', torch.cuda.get_device_name(torch.device('cuda:0')))

seed = 0


# In[ ]:


metadata = pd.read_csv(r"Desktop\コンペ\ISIC 2024 - Skin Cancer Detection with 3D-TBP\isic-2024-challenge\train-metadata.csv")
print(metadata["target"].value_counts())

kf = StratifiedGroupKFold(n_splits = 5, shuffle = True, random_state = seed)
for i, (tr_idx, val_idx) in enumerate(kf.split(metadata, metadata["target"], metadata["patient_id"])):
    metadata.loc[val_idx, "fold"] = i
    print(metadata[metadata["fold"] == i]["target"].value_counts())
gc.collect()
display(metadata)
# for i in range(0, 5):
#     temp = metadata[metadata["fold"] == i]
#     print(f'fold{i} shape:{temp.shape}')
#     print(temp["target_0"].value_counts())


# In[ ]:


class ISICDataset(Dataset):
    def __init__(self, df, over_sampling_rate, image_augment = True, folder_name = "Desktop\\コンペ\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\isic-2024-challenge\\train-image\\image\\"):
        self.df = df
        self.original_df = df
        self.image_augment = image_augment
        self.folder_name = folder_name
        self.over_sampling_rate = over_sampling_rate
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        target = self.df.iloc[index, 1].astype(np.float32)
        y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        
        image_id = self.df.iloc[index]["isic_id"]
        file_name = image_id + ".jpg"
        image = Image.open(self.folder_name + file_name)
        image = np.array(image, dtype = np.float32)
        image = self.__resize(image)/255
        
        if self.image_augment:
            image = self.__transform(image)
        
        x = self.__normalize(image)
        
        return x, y
    
    def __resize(self, image):
        resize = A.Compose([A.Resize(224, 224)])
        image = resize(image = image)["image"]
        return image
            
    def __normalize(self, image):
        normalize = A.Compose([A.Normalize(mean=[0., 0., 0.], std=[1, 1, 1], max_pixel_value=1.0), ToTensorV2()])
        image = normalize(image = image)["image"]
        return image
    
    def __transform(self, image):
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
#             A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 5)),
                A.Blur(blur_limit=(1, 5)),
                A.MotionBlur(blur_limit=(3, 5)),
                A.MedianBlur(blur_limit=(3, 5)),
                A.GaussNoise(var_limit=(5e-4, 1e-3), p=0.5)]),
            A.OneOf([
                A.OpticalDistortion(distort_limit=(-0.2, 0.2)),
                A.GridDistortion(num_steps=5, distort_limit=(-0.2, 0.2), p=0.5)]),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, border_mode=0, p=0.5),
#             A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            A.CoarseDropout(max_holes=8, max_height=int(224 * 0.05), max_width=int(224 * 0.05), min_holes=1, p=0.7),
#             A.CenterCrop(height=100, width=100, p=0.3)
        ])
        image = transform(image = image)["image"]
        return image
    
#     def undersampling(self, random_state):
#         negative_df = self.df[self.df.iloc[:, 1] == 0].sample(frac = 0.0618, random_state = random_state).reset_index(drop=True)
#         positive_df = self.df[self.df.iloc[:, 1] == 1].reset_index(drop=True)
#         self.df = pd.concat([positive_df, negative_df], axis = 0)
        
    def reset_df(self):
        self.df = self.original_df
        
    def oversampling(self, random_state):
        negative_df = self.df[self.df.iloc[:, 1] == 0].reset_index(drop=True)
        positive_df = self.df[self.df.iloc[:, 1] == 1].sample(frac = self.over_sampling_rate, replace=True, random_state = random_state).reset_index(drop=True)
        self.df = pd.concat([positive_df, negative_df], axis = 0)


# In[ ]:


# class GeM(nn.Module):
#     def __init__(self, p=3, eps=1e-6):
#         super(GeM, self).__init__()
#         self.p = nn.Parameter(torch.ones(1)*p)
#         self.eps = eps

#     def forward(self, x):
#         return self.gem(x, p=self.p, eps=self.eps)
        
#     def gem(self, x, p=3, eps=1e-6):
#         return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
#     def __repr__(self):
#         return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


# In[ ]:


class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes = 1, pretrained = True):
        super(ISICModel, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes = num_classes,
#             global_pool = "",
            in_chans = 3
        )
        in_features = self.model.num_features
#         self.pooling = GeM()
#         self.neck = nn.Sequential(
#             nn.Linear(in_features, 256),
#             nn.Dropout(0.5),
#             nn.SiLU(),
#                                  )
#         self.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        out = self.model(x)
#         x = self.pooling(x).flatten(1)
# #         x = self.neck(x)
#         out = self.head(x)
        return out


# In[ ]:


def train_one_epoch(model, optimizer, dataloader, criterion, device, use_cutmix):    
    train_loss = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for x, y in tqdm(dataloader, total=len(dataloader)):
        optimizer.zero_grad()
        x = x.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)
        if use_cutmix:
            new_x, y, new_y, lam = cutmix(x, y, 0.5)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                y_pred = model(new_x)
                loss = criterion(y_pred, new_y)
        else:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * x.shape[0]

    train_loss /= len(dataloader)
    
    return train_loss

def valid_one_epoch(model, dataloader, criterion, device):
    valid_loss = 0
    model.eval()
    labels = np.array(())
    preds = np.array(())
    for x, y in tqdm(dataloader, total=len(dataloader)):
        x = x.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)
        with torch.set_grad_enabled(False):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            valid_loss += loss.item() * x.shape[0]
            
        labels = np.concatenate([labels, y.view(-1).to('cpu').detach().numpy()])

        pred = F.sigmoid(y_pred).view(-1).to('cpu').detach().numpy().astype(np.float32)
        preds = np.concatenate([preds, pred])

    valid_loss /= len(dataloader)
    
    cm = confusion_matrix(labels, np.where(preds>=0.5, 1, 0))
    pauc = score(labels, preds)
    
    return valid_loss, cm, pauc, preds


# In[ ]:


def train_one_fold(model_name, df, fold, sampling_rate, batch_size_train, batch_size_valid, augmentation, epochs, early_stop, use_cutmix, criterion, save, ver, output_path, device):
    
    criterion.to(device)
    
    model = ISICModel(model_name = model_name, num_classes = 1, pretrained = True)
    model.to(device)
    
    train_ds = ISICDataset(df.loc[df["fold"] != fold], over_sampling_rate = sampling_rate, image_augment = augmentation, folder_name = "Desktop\\コンペ\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\isic-2024-challenge\\train-image\\image\\")
    train_loader = DataLoader(train_ds, batch_size = batch_size_train, shuffle=True, num_workers = 0, pin_memory = True)
    valid_ds = ISICDataset(df.loc[df["fold"] == fold], over_sampling_rate = 1, image_augment = False, folder_name = "Desktop\\コンペ\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\isic-2024-challenge\\train-image\\image\\")
    valid_loader = DataLoader(valid_ds, batch_size = batch_size_valid, shuffle=False, num_workers = 0, pin_memory = True)
    
    optimizer = optim.AdamW(params=model.parameters(), lr=2e-5, weight_decay=1e-2, eps=1e-8)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20, eta_min = 5e-8)
    
    best_loss = 1e+9
    best_epoch = 0
    best_score = 0
    
    for epoch in range(1, epochs + 1):
        
        train_ds.reset_df()
        train_ds.oversampling(random_state = epoch)
        
        train_loss = train_one_epoch(model, optimizer, train_loader, criterion, device, use_cutmix)
        print(f'fold {fold}, epoch {epoch}, train_loss = {train_loss}, lr = ' + str(optimizer.param_groups[0]['lr']))
        
        valid_loss, cm, pauc, _ = valid_one_epoch(model, valid_loader, criterion, device)
        print(f'fold {fold}, epoch {epoch}, valid_loss = {valid_loss}, pAUC = {pauc}, cofusion_matrix = {cm}')
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_score = pauc
            
            if save:
                torch.save(model.to('cpu').state_dict(), output_path +  f'fold_{fold}_{model.model_name}_ver{ver}.pth')
                model.to(device)
                print(f"saved fold_{fold}_{model.model_name}_ver{ver}")
        
        if not early_stop == None:
            if (epoch - best_epoch > early_stop) & (valid_loss > best_loss):
                print("early stopped")
                break

        scheduler.step()

    print(f"fold:{fold}, best epoch:{best_epoch}, best loss:{best_loss}")

    torch.cuda.empty_cache()
    gc.collect()
    return best_score


# In[ ]:


folds = [0.0, 4.0]
scores = []
for fold in folds:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    scores.append(train_one_fold(model_name = "tf_efficientnet_b0", 
                                 df = metadata, 
                                 fold = fold, 
                                 sampling_rate = 20, 
                                 batch_size_train = 64, 
                                 batch_size_valid = 64, 
                                 augmentation = True, 
                                 epochs = 71,
                                 early_stop = 5,
                                 use_cutmix = False,
                                 criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([70])),
                                 save = False, 
                                 ver = "_", 
                                 output_path = "Desktop\\コンペ\\ISIC 2024 - Skin Cancer Detection with 3D-TBP\\isic-2024-challenge\\", 
                                 device = device))
print(score)


# In[ ]:


torch.cuda.empty_cache()
gc.collect()


# In[ ]:




