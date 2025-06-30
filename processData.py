import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.datasets import load_breast_cancer
import os
import torch
import torch.nn as nn
import h5py
import torch.utils.data as Data
import pandas as pd
import random
import glob


use_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
keys = ["Fpz-Cz", "Pz-Oz", "label"]
result_path = "./result/"
##Saving scaler
'''
scaler_path = './result/scaler/'
scaler_files = os.listdir(scaler_path)
savepseudofile = './pseudodata/'
'''

def saveLabelFile(file_name, contents):
    df = pd.DataFrame(contents)
    df.to_csv(file_name, index=None, header=None)

def getEEGData_group(h5file, filesname, channel, seq_len=10):
    # random.shuffle(filesname)
    data = np.empty(shape=[0, 3000])
    labels = np.empty(shape=[0, 1])
    index = np.array([])
    num = 0
    # random.shuffle(filesname)
    for filename in filesname:
        with h5py.File(h5file + filename, 'r') as fileh5:
            '''
            data_temp, label_temp = get_seq_data(fileh5[keys[channel]][:], fileh5[keys[2]][:], seq_len)
            data = np.concatenate((data, data_temp), axis=0)
            labels = np.concatenate((labels, label_temp), axis=0)
            index.append([num] * data_temp.shape[0])
            '''
            data_temp = fileh5[keys[2]][:]
            data = np.concatenate((data, fileh5[keys[channel]][:]), axis=0)
            labels = np.concatenate((labels, fileh5[keys[2]][:]), axis=0)
            index = np.append(index, np.array([num] * data_temp.shape[0]), axis=0)
            num += 1
    data = (torch.from_numpy(data)).type('torch.FloatTensor')
    labels = (torch.from_numpy(labels)).type('torch.LongTensor')
    labels = labels.squeeze(dim=1)
    return data, labels, index


def load_and_concat_npz(folder,
                        x_key='x',
                        y_key='y',
                        sort_files=True,
                        verbose=False):
    index = np.array([])
    num = 0
    pattern = os.path.join(folder, '*.npz')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")
    if sort_files:
        files.sort()

    xs, ys = [], []
    for i, fpath in enumerate(files, 1):
        with np.load(fpath, mmap_mode=None) as data:
            if x_key not in data or y_key not in data:
                raise KeyError(f"{fpath!r} 中缺少 '{x_key}' 或 '{y_key}'")
            xs.append(data[x_key])
            ys.append(data[y_key])
            index = np.append(index, np.array([num] * data[x_key].shape[0]), axis=0)
            num += 1
        if verbose:
            print(f"[{i}/{len(files)}] loaded {os.path.basename(fpath)}, "
                  f"x shape={xs[-1].shape}, y shape={ys[-1].shape}")

    # 按第 0 维度拼接
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    return x_all, y_all, index

def getEEGData_withoutSeq(h5file, filesname, channel):
    data = np.empty(shape=[0, 3000])
    labels = np.empty(shape=[0, 1])
    # random.shuffle(filesname)
    for filename in filesname:
        with h5py.File(h5file + filename, 'r') as fileh5:
            data = np.concatenate((data, fileh5[keys[channel]][:]), axis=0)
            labels = np.concatenate((labels, fileh5[keys[2]][:]), axis=0)
    data = (torch.from_numpy(data)).type('torch.FloatTensor')
    labels = (torch.from_numpy(labels)).type('torch.LongTensor')
    labels = labels.squeeze(dim=1)
    return data, labels

def get_seq_data(data, label, seq_len=10):
    datas = []
    labels = []
    for i in range(len(data) - seq_len):
        datas.append(np.array(data[i:i + seq_len, :]))
        # labels.append(label[i + seq_len - 1])
        # labels.append(label[(i + seq_len) // 2])
        labels.append(label[i])
    return np.array(datas), np.array(labels)

###Normalizing and return t he normalized data and standardscaler
def standardScalerData(standardscaler, x_data):
    standardscaler.fit(x_data)
    x_standard = standardscaler.transform(x_data)
    return torch.from_numpy(x_standard), standardscaler

def cutData(x_data, y_data, size):
    len = x_data.shape[0] // size * size
    x_data = x_data[:len, :]
    y_data = y_data[:len]

    return x_data, y_data

##Reshaping the data as a batch
def shuffleData(x_data, y_data, size):

    x_data = x_data.reshape(-1, size, 3000)
    y_data = y_data.reshape(-1, size, 1)

    return x_data, y_data

##Reshaping the data according to a batch
def noshuffleData(x_data, y_data):
    x_data = x_data.reshape(-1, 3000)
    y_data = y_data.reshape(-1, 1)

    return x_data, y_data

def checkDataset(label):
    count = [0] * 5
    for item in enumerate(label,0):
        count[int(item[1])] += 1
    return count


class Customized_slide_window_Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, window_size=5):
        self.data = x
        self.label = y
        self.window_size = window_size

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.window_size
        window_data = self.data[start_index:end_index]
        window_label = self.label[(start_index + end_index) // 2]
        return window_data, window_label

    def __len__(self):
        return len(self.data) - self.window_size + 1
