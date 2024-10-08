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
import timm
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import colorednoise as cn
from albumentations import BasicTransform
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
        white_noise = white_noise * mean_volume
        audio = audio + white_noise
        return audio, sr
    
class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(PinkNoise, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        audio, sr = data
        pink_noise = cn.powerlaw_psd_gaussian(1, len(audio))
        mean_volume = np.sqrt(np.mean(audio ** 2))
        pink_noise = pink_noise * mean_volume
        audio = audio + pink_noise
        return audio, sr

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


# In[ ]:


for index, row in metadata.iterrows():
    strings = extract_strings(row["secondary_labels"])
    label = row["primary_label"]
    if any(element in strings for element in ["magrob", "indwhe1", "lotshr1", "asfblu1", "orhthr1", "bltmun1"]):
        for element in ["magrob", "indwhe1", "lotshr1", "asfblu1", "orhthr1", "bltmun1"]:
            if element in strings:
                strings.remove(element)
    if len(strings) == 0:
        pass
    elif len(strings) == 1:
        metadata.loc[index, label] = 0.7
        temp = strings[0]
        metadata.loc[index, temp] = 0.3
    else:
        metadata.loc[index, label] = 0.5
        Q = (1-0.5) / len(strings)
        for str in strings:
            metadata.loc[index, str] = Q


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
        if self.mode == "train":
            choice = np.random.randint(0, 3)
            if choice == 0:
                audio = audio[:self.segment_duration * sr]
            elif choice == 1:
                start = np.random.uniform(0, duration - self.segment_duration)
                start_time = int(start * sr)
                end_time = int((start + self.segment_duration) * sr)
                audio = audio[start_time:end_time]
            else:
                audio = audio[-self.segment_duration * sr:]
        else:
            audio = audio[:self.segment_duration * sr]
        
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
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        
        return X, y
    
    def __audiotransform(self, data):
        transform = A.Compose([
            PitchShift(p=0.5),
            TimeShift(p=0.5),
            WhiteNoise(p=0.5),
            PinkNoise(p=0.5)
        ])
        audio, sr = transform(data=data)['data']
        return audio, sr
    
    def __imagetransform(self, image):
        transform = A.Compose([
            FreqMasking(p=0.5),
            TimeMasking(p=0.5)
        ])
        image = transform(data=image)['data']
        return image


# In[ ]:


class BirdModel(nn.Module):
    def __init__(self, model_name, num_classes = 182, pretrained = True):
        super(BirdModel, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes = num_classes,
            in_chans = 1
        )
        
    def forward(self, x):
        out = self.model(x)
        return out


# In[ ]:


def model_training(model_name, metadata, kf, batch_size_train, batch_size_valid, epochs, lr, device, output_path, early_stop, ver, debug):
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    for i, (tr_idx, va_idx) in enumerate(kf.split(metadata, metadata["primary_label"])):
        model = BirdModel(model_name = model_name, num_classes = 182, pretrained = True)       
        model.to(device)
        
        optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-2, eps=1e-8)
        
        train_ds = TrainBirdDataset(metadata.iloc[tr_idx], mode = "train", segment_duration = 5, audio_augment = True, image_augment = True, folder_name = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\train_audio\\")
        train_loader = DataLoader(train_ds, batch_size = batch_size_train, shuffle=True, num_workers = 0, pin_memory = True)
        valid_ds = TrainBirdDataset(metadata.iloc[va_idx], mode='valid', segment_duration = 5, audio_augment = False, image_augment = False, folder_name = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\train_audio\\")
        valid_loader = DataLoader(valid_ds, batch_size = batch_size_valid, shuffle=False, num_workers = 0, pin_memory = True)
        
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr = 5e-4, total_steps = epochs, pct_start = 0.2, anneal_strategy = "cos", three_phase = True)
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
                    y_pred = model(x)
                    loss = criterion(F.log_softmax(y_pred, dim=1), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e7)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * x.shape[0]

            train_loss /= len(train_loader)
            print(f'fold {i}, epoch {epoch}, train_loss = {train_loss}')
            
            model.eval()
            labels = torch.tensor(())
            preds = torch.tensor(())
            for x, y in valid_loader:
                x = x.to(device, non_blocking = True)
                y = y.to(device, non_blocking = True)
                with torch.set_grad_enabled(False):
                    y_pred = model(x)
                    loss = criterion(F.log_softmax(y_pred, dim=1), y)
                    valid_loss += loss.item() * x.shape[0]
                    
                    _, correct_label = torch.max(y, 1)
                    labels = torch.cat((labels, correct_label.to('cpu')), dim = 0)
                    
                    _, pred_label = torch.max(y_pred, 1)
                    preds = torch.cat((preds, pred_label.to('cpu')), dim = 0)
            valid_loss /= len(valid_loader)
            print(f'fold {i}, epoch {epoch}, valid_loss = {valid_loss}, Accuracy = {accuracy_score(labels, preds)}')
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                torch.save(model.to('cpu').state_dict(), output_path +  f'fold_{i}_{model.model_name}_ver{ver}.pth')
                model.to(device)
                print(f"saved fold_{i}_{model.model_name}_ver{ver}")

            if (epoch - best_epoch > early_stop) & (valid_loss > best_loss):
                print("early stopped")
                break
            
            scheduler.step()
                
        print(f"fold:{i}, best epoch:{best_epoch}, best loss:{best_loss}")
        if debug == True and i == 0:
            break
    del model, optimizer, train_ds, train_loader, valid_ds, valid_loader
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:


ver = "1_3"
batch_size_train = 114
batch_size_valid = 114
epochs = 50
lr = 1e-3
seed = 0
early_stop = 10
model_name = "resnet34d"
output_path = "Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\"
debug = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
kf = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = True)
model_training(model_name, metadata, kf, batch_size_train, batch_size_valid, epochs, lr, device, output_path, early_stop, ver, debug)


# In[ ]:


#読み込み
audio, sr = librosa.load("Desktop\\コンペ\\BirdCLEF 2024\\birdclef-2024\\train_audio\\asbfly\\XC164848.ogg", sr = None)

#元の音声の長さ
duration = len(audio) / sr
print(f'元の音声の長さ：{duration}')

#最初の5秒
audio = audio[:5 * sr]
duration = len(audio) / sr
print(f'5秒の音声の長さ：{duration}')

#音声Augmentationの実験場所


#音声の確認
display(Audio(audio, rate =sr))
display(Audio(audio_, rate =sr))

#メルスペクトログラム
S = librosa.feature.melspectrogram(y = audio_, sr = sr, n_mels = 256, n_fft = 2048, hop_length = len(audio) // 255)
S = librosa.amplitude_to_db(S, ref = np.max)
print(f'メルスペクトログラムのShape：{S.shape}')

#標準化
image_mean = np.mean(S)
image_std = np.std(S)
print(f'平均値:{image_mean}, 標準偏差:{image_std}')
S = (S - image_mean) / (image_std + 1e-6)

#正規化
image_max = np.max(S)
image_min = np.min(S)
print(f'最大値:{image_max}, 最小値:{image_min}')
S = (S-image_min) / (image_max - image_min)


#メルスペクトログラムの表示
librosa.display.specshow(S, x_axis = "time", y_axis = "mel", sr = sr)
plt.colorbar()
plt.show()


# In[ ]:




