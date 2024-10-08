#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_rows = 200
pd.options.display.max_columns = 100
import numpy as np
np.set_printoptions(threshold=100)
from IPython.display import Audio
import os
import glob
import random
import gc
import warnings
warnings.filterwarnings('ignore')
import timm
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import colorednoise as cn
from albumentations import BasicTransform
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class AudioTransform(BasicTransform):
    """Transform for Audio task"""
    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params
    
    
class FreqMasking(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(FreqMasking, self).__init__(always_apply, p)
    
    def apply(self, spec, **params):
        F = spec.shape[0]
        for _ in range(2):
            freq_percentage = random.uniform(0.0, 0.1)
            num_freqs_to_mask = int(freq_percentage * F)
            f0 = np.random.uniform(low=0.0, high=F - num_freqs_to_mask)
            f0 = int(f0)
            spec[f0:f0 + num_freqs_to_mask, :] = 0
        return spec

class TimeMasking(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(TimeMasking, self).__init__(always_apply, p)
    
    def apply(self, spec, **params):
        T = spec.shape[1]
        for _ in range(2):
            time_percentage = random.uniform(0.0, 0.15)
            num_times_to_mask = int(time_percentage * T)
            t0 = np.random.uniform(low=0.0, high=T - num_times_to_mask)
            t0 = int(t0)
            spec[:, t0:t0 + num_times_to_mask] = 0
        return spec

class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(PitchShift, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        audio, sr = data
        n_steps = np.random.randint(-3, 3)
        audio = librosa.effects.pitch_shift(y = audio, sr = sr, n_steps = n_steps)
        return audio, sr
    
class TimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(TimeShift, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        audio, sr = data
        shift = np.random.randint(-3, 3)*sr
        audio = np.roll(audio, shift)
        return audio, sr
    
class WhiteNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(WhiteNoise, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        audio, sr = data
        white_noise = np.random.randn(len(audio))
        mean_volume = np.sqrt(np.mean(audio ** 2))
        white_noise = white_noise * mean_volume * 0.8
        audio = audio + white_noise
        return audio, sr
    
class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(PinkNoise, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        audio, sr = data
        pink_noise = cn.powerlaw_psd_gaussian(1, len(audio))
        mean_volume = np.sqrt(np.mean(audio ** 2))
        pink_noise = pink_noise * mean_volume * 0.8
        audio = audio + pink_noise
        return audio, sr
    
def mixup_training(criterion, X, y, model, alpha = 0.5):
    lam = np.random.beta(alpha, alpha)
    batch_size = X.shape[0]
    index = torch.randperm(batch_size)
    mixed_X = lam * X + (1 - lam) * X[index, :]
    y_a, y_b = y, y[index]
    
    y_pred = model(mixed_X)
    
    return lam * criterion(y_pred, y_a) + (1 - lam) * criterion(y_pred, y_b)
    

def extract_strings(string):
    if string == "[]":
        return []

    string = string.strip("[]")
    strings = string.split(", ")

    return [s.strip("'") for s in strings]

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)', torch.cuda.get_device_name(torch.device('cuda:0')))


# In[ ]:


metadata = pd.read_csv("Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\train_metadata.csv")
print(metadata.shape)
print(len(metadata["primary_label"].unique()))
print(metadata["primary_label"].value_counts())


# In[ ]:


temp = pd.get_dummies(metadata["primary_label"], columns = ["primary_label"], dtype = float, prefix='', prefix_sep='')
metadata = pd.concat([metadata, temp], axis = 1)
for index, row in metadata.iterrows():
    strings = extract_strings(row["secondary_labels"])
    label = row["primary_label"]
    if any(element in strings for element in ["magrob", "indwhe1", "lotshr1", "asfblu1", "orhthr1", "bltmun1"]):
        for element in ["magrob", "indwhe1", "lotshr1", "asfblu1", "orhthr1", "bltmun1"]:
            if element in strings:
                strings.remove(element)


# In[ ]:


class TrainBirdDataset(Dataset):
    def __init__(self, df, mode = "train", segment_duration = 5, audio_augment = True, image_augment = True, folder_name = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\train_audio\\"):
        self.df = df
        self.mode = mode
        self.segment_duration = segment_duration
        self.audio_augment = audio_augment
        self.image_augment = image_augment
        self.folder_name = folder_name
        
    def __len__(self):
        return len(self.df)
    
    def __getaudio(self, index):
        file_name = self.df.iloc[index]["filename"]
        audio, sr = librosa.load(self.folder_name + file_name, sr = None)
        duration = len(audio) / sr
        if duration < self.segment_duration:
            repeat = int(self.segment_duration / duration) + 1
            audio = np.tile(audio, repeat)
            duration = len(audio) / sr
        
        start = np.random.uniform(0, duration - self.segment_duration)
        start_time = int(start * sr)
        end_time = int((start + self.segment_duration) * sr)
        audio = audio[start_time:end_time]

        return audio, sr
            
    def __getimage(self, index):
        audio, sr = self.__getaudio(index)
        if self.mode == "train" and self.audio_augment == True:
            data = audio, sr
            audio, sr = self.__audiotransform(data)
        S = librosa.feature.melspectrogram(y = audio, sr = sr, n_mels = 256, n_fft = 2048, hop_length = len(audio) // 255)
        S = librosa.amplitude_to_db(S, ref = np.max)
        image_mean = np.mean(S)
        image_std = np.std(S)
        S = (S - image_mean) / (image_std + 1e-6)
        image_max = np.max(S)
        image_min = np.min(S)
        S = (S-image_min) / (image_max - image_min + 1e-6)
 
        return S
    
    def __getitem__(self, index):
        X = self.__getimage(index)
        if self.mode == "train" and self.image_augment == True:
            X = self.__imagetransform(X)
        y = self.df.iloc[index, -182:].to_numpy().astype(np.float32)
        X = X.reshape(1, X.shape[0], X.shape[1]).astype(np.float32)
        X = np.nan_to_num(X)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        
        return X, y
    
    def __audiotransform(self, data):
        transform = A.Compose([
            PitchShift(p=0.5),
            TimeShift(p=0.5),
            A.OneOf([WhiteNoise(p=0.5),
                     PinkNoise(p=0.5)], p=0.5)
        ])
        audio, sr = transform(data=data)['data']
        return audio, sr
    
    def __imagetransform(self, image):
        transform = A.Compose([
            FreqMasking(p=0.5),
            TimeMasking(p=0.5)
        ])
        image = transform(data=image)['data']
        
        image_max = np.max(image)
        image_min = np.min(image)
        image = (image-image_min) / (image_max - image_min + 1e-6)
        return image


# In[ ]:


#https://www.kaggle.com/code/hidehisaarai1213/pytorch-training-birdclef2021-starter
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)
    
def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()
        
def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)
    return output

# def gem(x: torch.Tensor, p=3, eps=1e-6):
#     return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


# class GeM(nn.Module):
#     def __init__(self, p=3, eps=1e-6):
#         super().__init__()
#         self.p = nn.Parameter(torch.ones(1) * p)
#         self.eps = eps

#     def forward(self, x):
#         return gem(x, p=self.p, eps=self.eps)

#     def __repr__(self):
#         return self.__class__.__name__ + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class AttBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()
        self.activation = activation
        self.att = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm1d(182)
        self.bn2 = nn.BatchNorm1d(182)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        x_ = self.att(x)
        x_ = self.bn1(x_)
        x_ = torch.tanh(x_)
        norm_att = torch.softmax(x_, dim=-1)
        
        x_ = self.cla(x)
        x_ = self.bn2(x_)
        cla = self.nonlinear_transform(x_)
        
        y = torch.sum(norm_att * cla, dim=2)
        return y, norm_att, cla
    
    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x+1e-8)


# In[ ]:


class BirdModel(nn.Module):
    def __init__(self, model_name, 
                 timm_classes: int, 
                 classes: int, 
                 pretrained = True):
        super(BirdModel, self).__init__()
        self.model_name = model_name
        self.classes = classes
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes = timm_classes,
            global_pool = "",
            in_chans = 1
        )
        in_features = self.backbone.num_features
            
        self.bn0 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlock(in_features, self.classes, activation="sigmoid")

        self.init_weight()
    
    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)
        
    def forward(self, x):
        #input (bs, c, freq, time)
#         print(f'input > bs, c, freq, time:{x.shape}')
        frames = x.shape[3]
        x = x.transpose(1, 2) #(bs, freq, c, time)
#         print(f'tanspose1 > bs, freq, c, time:{x.shape}')
        x = self.bn0(x)
#         print(f'bn > bs, freq, c, time:{x.shape}')
        x = x.transpose(1, 2) #(bs, c, freq, time)
#         print(f'tanspose2 > bs, c, freq, time:{x.shape}')
        x = self.backbone(x) #(bs, c, freq, time)
#         print(f'encoder > bs, c, freq, time:{x.shape}')
        x = torch.mean(x, dim = 2) #(bs, c, time)
        
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2) #(bs, time, c)
        x = self.fc1(x)
        x = F.relu_(x)
        x = x.transpose(1, 2) #(bs, c, time)
        x = F.dropout(x, p=0.2, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        #(bs, class_num) (bs, class_num, c) (bs, class_num, c)

        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2) #(bs, class_num)
        
        
        segmentwise_output = segmentwise_output.transpose(1, 2) #(bs, c, class_num)
        interpolate_ratio = frames // segmentwise_output.size(1)
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames)
        
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2) #(bs, c, class_num)
        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames)
        
        output_dict = {"framewise_output": framewise_output,
                       "segmentwise_output": segmentwise_output,
                       "logit": logit,
                       "framewise_logit": framewise_logit,
                       "clipwise_output": clipwise_output,
                       "norm_att": norm_att}
        
        return output_dict


# In[ ]:


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        bce_loss = criterion(preds+1e-8, targets)
        loss = bce_loss.mean()
        return loss
    
class BCE3WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.bce = BCELoss()

        self.weights = weights

    def forward(self, input, target):
        input1 = input["clipwise_output"]
        input2 = input["logit"]
        target = target.float()

        framewise_output = input["framewise_output"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss1 = self.bce(input1, target)
        loss2 = self.bce(input2, target)
        loss3 = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * loss1 + self.weights[1] * loss2 + self.weights[2] * loss2
    
class BCE3WayLossLabelSmooth(nn.Module):
    def __init__(self, weights=[1, 1, 1], smooth_eps = 0.025, class_weights=None):
        super().__init__()
        self.bce = BCELoss()
        self.weights = weights
        self.smooth_eps = smooth_eps
        

    def forward(self, input, target):
        input1 = input["clipwise_output"]
        input2 = input["logit"]
        target = torch.clamp(target.float(), self.smooth_eps, 1.0 - self.smooth_eps)

        framewise_output = input["framewise_output"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss1 = self.bce(input1, target)
        loss2 = self.bce(input2, target)
        loss3 = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * loss1 + self.weights[1] * loss2 + self.weights[2] * loss2


# In[ ]:


def model_training(model_name, metadata, kf, batch_size_train, batch_size_valid, epochs, lr,
                   device, output_path, early_stop, ver, debug):
    train_criterion = BCE3WayLossLabelSmooth(weights = [1, 1, 1])
    train_criterion.to(device)
    valid_criterion = BCE3WayLoss(weights = [1, 1, 1])
    valid_criterion.to(device)
    for i, (tr_idx, va_idx) in enumerate(kf.split(metadata, metadata["primary_label"])):
        temp = metadata.iloc[tr_idx]["primary_label"].value_counts()
        over_sample_class = temp[temp <= 40].index
        
        data_to_resample = metadata.iloc[tr_idx][metadata["primary_label"].isin(over_sample_class)]
        original_number_data = metadata.iloc[tr_idx][~metadata["primary_label"].isin(over_sample_class)]
        
        ros = RandomOverSampler(random_state = 0)
        resampled_features, resampled_target = ros.fit_resample(data_to_resample.drop(["primary_label"], axis = 1), data_to_resample["primary_label"])
        
        over_sampled_data = pd.concat([resampled_features, resampled_target], axis = 1)
        
        new_data = pd.concat([original_number_data, over_sampled_data], axis = 0)
            
        model = BirdModel(model_name = model_name, timm_classes = 0, classes = 182, pretrained = True)
        model.to(device)
        
        optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-2, eps=1e-8)
        
        train_ds = TrainBirdDataset(new_data, mode = "train", segment_duration = 20, audio_augment = True, image_augment = True, folder_name = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\train_audio\\")
        train_loader = DataLoader(train_ds, batch_size = batch_size_train, shuffle=True, num_workers = 0, pin_memory = True)
        
        valid_ds = TrainBirdDataset(metadata.iloc[va_idx], mode='valid', segment_duration = 20, audio_augment = False, image_augment = False, folder_name = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\train_audio\\")
        valid_loader = DataLoader(valid_ds, batch_size = batch_size_valid, shuffle=False, num_workers = 0, pin_memory = True)
        
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr = 1e-5, max_lr = 1e-3, step_size_up = 5, mode = "exp_range", gamma = 0.93, cycle_momentum = False)
        best_loss = 1e+9
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_loss = 0
            valid_loss = 0
            model.train()
            scaler = torch.cuda.amp.GradScaler(enabled = True)
            for j, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
                optimizer.zero_grad()
                x = x.to(device, non_blocking = True)
                y = y.to(device, non_blocking = True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    loss = mixup_training(criterion = train_criterion, X = x, y = y, model = model, alpha = 0.3)
#                     y_pred = model(x)
#                     loss = train_criterion(y_pred, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * x.shape[0]
            train_loss /= len(train_loader)
            print(f'fold {i}, epoch {epoch}, train_loss = {train_loss}')


            model.eval()
            labels = torch.tensor(())
            preds1 = torch.tensor(())
            preds2 = torch.tensor(())
            preds3 = torch.tensor(())
            pred_probas = torch.tensor(())
            for x, y in valid_loader:
                x = x.to(device, non_blocking = True)
                y = y.to(device, non_blocking = True)
                with torch.set_grad_enabled(False):
                    y_pred = model(x)
                    loss = valid_criterion(y_pred, y)
                    valid_loss += loss.item() * x.shape[0]

                    _, correct_label = torch.max(y, 1)
                    labels = torch.cat((labels, correct_label.to('cpu')), dim = 0)

                    pred_proba, pred_label1 = torch.max(F.softmax(y_pred["logit"], dim = 1), dim = 1)
                    preds1 = torch.cat((preds1, pred_label1.to('cpu')), dim = 0)
                    pred_probas = torch.cat((pred_probas, pred_proba.to('cpu')), dim = 0)

                    _, pred_label2 = torch.max(y_pred["clipwise_output"], dim = 1)
                    preds2 = torch.cat((preds2, pred_label2.to('cpu')), dim = 0)

                    p, _ = torch.max(y_pred["framewise_output"], dim = 1)
                    _, pred_label3 = torch.max(p, 1)
                    preds3 = torch.cat((preds3, pred_label3.to('cpu')), dim = 0)

            valid_loss /= len(valid_loader)
            print(f'fold {i}, epoch {epoch}, valid_loss = {valid_loss}, logitAccuracy1 = {accuracy_score(labels, preds1)}, clipAccuracy2 = {accuracy_score(labels, preds2)}, frameAccuracy3 = {accuracy_score(labels, preds3)}')
            print(f'labels:{labels[:50]}')
            print(f'preds:{preds1[:50]}')
            print(f'probas:{pred_probas[:50]}')

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                torch.save(model.to('cpu').state_dict(), output_path +  f'fold_{i}_{model.model_name}_ver{ver}.pth')
                model.to(device)
                print(f"saved fold_{i}_{model.model_name}_ver{ver}")

            if (epoch - best_epoch > early_stop) & (valid_loss > best_loss):
                print("early stopped")
                break
                
            if epoch == 40:
                break

            scheduler.step()

        print(f"fold:{i}, best epoch:{best_epoch}, best loss:{best_loss}")

    del model, optimizer, train_ds, train_loader, valid_ds, valid_loader
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:


ver = "1_2_1"
batch_size_train = 114
batch_size_valid = 114
epochs = 50
lr = 1e-3
seed = 0
early_stop = 8
model_name = "eca_nfnet_l0"
output_path = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\"
debug = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
kf = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = True)
model_training(model_name, metadata, kf, batch_size_train, batch_size_valid, epochs, lr, 
               device, output_path, early_stop, ver, debug)


# In[ ]:


def just_model_training(model_name, metadata, batch_size_train, epochs, lr, device, output_path, ver):
    train_criterion = BCE3WayLossLabelSmooth(weights = [1, 1, 1])
    train_criterion.to(device)
    
    temp = metadata["primary_label"].value_counts()
    over_sample_class = temp[temp <= 51].index

    data_to_resample = metadata[metadata["primary_label"].isin(over_sample_class)]
    original_number_data = metadata[~metadata["primary_label"].isin(over_sample_class)]
    
    randomstate = np.random.randint(1, 100)
    ros = RandomOverSampler(random_state = randomstate)
    resampled_features, resampled_target = ros.fit_resample(data_to_resample.drop(["primary_label"], axis = 1), data_to_resample["primary_label"])

    over_sampled_data = pd.concat([resampled_features, resampled_target], axis = 1)

    new_data = pd.concat([original_number_data, over_sampled_data], axis = 0)

    model = BirdModel(model_name = model_name, timm_classes = 0, classes = 182, pretrained = True)
    model.to(device)

    optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-2, eps=1e-8)

    train_ds = TrainBirdDataset(new_data, mode = "train", segment_duration = 20, audio_augment = True, image_augment = True, folder_name = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\train_audio\\")
    train_loader = DataLoader(train_ds, batch_size = batch_size_train, shuffle=True, num_workers = 0, pin_memory = True)

    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr = 1e-5, max_lr = 1e-3, step_size_up = 5, mode = "exp_range", gamma = 0.93, cycle_momentum = False)
    
    for epoch in range(1, epochs + 1):
        train_loss = 0
        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled = True)
        for j, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            x = x.to(device, non_blocking = True)
            y = y.to(device, non_blocking = True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss = mixup_training(criterion = train_criterion, X = x, y = y, model = model, alpha = 0.3)

                if torch.isnan(loss):
                    print("detected Nan")
                    break
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.shape[0]
        train_loss /= len(train_loader)
        print(f'epoch {epoch}, train_loss = {train_loss}')
        
        if epoch == 40:
            torch.save(model.to('cpu').state_dict(), output_path +  f'{model.model_name}_ver{ver}.pth')
            model.to(device)
            print(f"saved {model.model_name}_ver{ver}")
            break

        scheduler.step()

    del model, optimizer, train_ds, train_loader
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:


ver = "3_1"
batch_size_train = 114
epochs = 50
lr = 1e-3
seed = 0
early_stop = 8
model_name = "eca_nfnet_l0"
output_path = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\"
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
just_model_training(model_name, metadata, batch_size_train, epochs, lr, device, output_path, ver)


# In[ ]:


ver = "3_1"
batch_size_train = 114
epochs = 50
lr = 1e-3
seed = 0
early_stop = 8
model_name = "tf_efficientnet_b0"
output_path = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\"
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
just_model_training(model_name, metadata, batch_size_train, epochs, lr, device, output_path, ver)


# In[ ]:


def just_efficientnet_training(model_name, metadata, batch_size_train, epochs, lr, device, output_path, ver):
    train_criterion = BCE3WayLossLabelSmooth(weights = [1, 1, 1], smooth_eps = 0.0001)
    train_criterion.to(device)
    
    temp = metadata["primary_label"].value_counts()
    over_sample_class = temp[temp <= 51].index

    data_to_resample = metadata[metadata["primary_label"].isin(over_sample_class)]
    original_number_data = metadata[~metadata["primary_label"].isin(over_sample_class)]
    
    randomstate = np.random.randint(1, 100)
    ros = RandomOverSampler(random_state = randomstate)
    resampled_features, resampled_target = ros.fit_resample(data_to_resample.drop(["primary_label"], axis = 1), data_to_resample["primary_label"])

    over_sampled_data = pd.concat([resampled_features, resampled_target], axis = 1)

    new_data = pd.concat([original_number_data, over_sampled_data], axis = 0)

    model = BirdModel(model_name = model_name, timm_classes = 0, classes = 182, pretrained = True)
    model.load_state_dict(torch.load("Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\tf_efficientnet_b0_ver3_1.pth", map_location = torch.device("cpu")))
    model.to(device)

    optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-2, eps=1e-8)

    train_ds = TrainBirdDataset(new_data, mode = "train", segment_duration = 20, audio_augment = True, image_augment = True, folder_name = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\train_audio\\")
    train_loader = DataLoader(train_ds, batch_size = batch_size_train, shuffle=True, num_workers = 0, pin_memory = True)

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, total_steps = epochs, pct_start = 0.28, anneal_strategy = 'cos', three_phase = True, final_div_factor = 50)
    
    best_loss = 86.5749936
    
    for epoch in range(1, epochs + 1):
        train_loss = 0
        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled = True)
        for j, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            x = x.to(device, non_blocking = True)
            y = y.to(device, non_blocking = True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss = mixup_training(criterion = train_criterion, X = x, y = y, model = model, alpha = 0.3)

                if torch.isnan(loss):
                    print("detected Nan")
                    break
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.shape[0]
        train_loss /= len(train_loader)
        print(f'epoch {epoch}, train_loss = {train_loss}')
        
        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.to('cpu').state_dict(), output_path +  f'{model.model_name}_ver{ver}.pth')
            model.to(device)
            print(f"saved {model.model_name}_ver{ver}")
        
        scheduler.step()

    del model, optimizer, train_ds, train_loader
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:


ver = "3_2"
batch_size_train = 114
epochs = 15
lr = 1e-3
seed = 1
early_stop = 8
model_name = "tf_efficientnet_b0"
output_path = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\"
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
just_efficientnet_training(model_name, metadata, batch_size_train, epochs, lr, device, output_path, ver)


# In[ ]:




