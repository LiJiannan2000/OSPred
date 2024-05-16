import os
import pickle
import random
from itertools import islice

import numpy as np
from core.config import config
from collections import OrderedDict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from torch.utils.data import Dataset, DataLoader
import torch
import SimpleITK as sitk


class Dataset3D_itemread(Dataset):
    def __init__(self, data, patch_size, input_size):
        super(Dataset3D_itemread, self).__init__()
        self._data = data
        self.input_size = input_size
        self.patch_size = patch_size
        print(self.patch_size)

        sels = list(self._data.keys())
        self.datas, self.datas_def, self.deaths, self.ostimes, self.info = [], [], [], [], []
        for name in sels:
            data = np.load(self._data[name]['path'])['data']
            data_def = np.load(self._data[name]['path1'])['data']
            death = np.load(self._data[name]['path'])['data_death']
            age = np.load(self._data[name]['path'])['data_age']
            tumorpos = np.load(self._data[name]['path'])['data_tumorpos']
            if death == 2:
                death = 0
            else:
                death = int(death)
            ostime = np.load(self._data[name]['path'])['data_ostime']
            ostime = ostime.astype(np.float32)
            info = np.concatenate(([age], tumorpos))
            info = info.astype(np.float32)

            shape = np.array(data.shape[1:])  # slice shape，(x, y, z)
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]),
                                 (pad_left[2], pad_right[2])))

            images = data[:self.input_size]

            # intensity normalization using z-score
            data_def = (data_def - data_def.mean()) / (data_def.std() + 1e-8)

            label = data[-1:]
            self.datas.append(images)
            self.datas_def.append(data_def)
            self.deaths.append(death)
            self.ostimes.append(ostime)
            self.info.append(info)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return {'data': self.datas[index], 'data_def': self.datas_def[index], 'death': self.deaths[index], 'ostime': self.ostimes[index], 'info': self.info[index]}


class Dataset3D_itemread_BraTS20(Dataset):
    def __init__(self, data, patch_size):
        super(Dataset3D_itemread_BraTS20, self).__init__()
        self._data = data
        self.patch_size = patch_size
        print(self.patch_size)

        sels = list(self._data.keys())
        self.datas, self.datas_def, self.deaths, self.ostimes, self.info = [], [], [], [], []
        for name in sels:
            data = np.load(self._data[name]['path'])['data']
            data_def = np.load(self._data[name]['path1'])['data']
            death = np.load(self._data[name]['path'])['data1']
            ostime = np.load(self._data[name]['path'])['data_ostime']
            age = np.load(self._data[name]['path'])['data_age']
            tumorpos = np.load(self._data[name]['path'])['data_tumorpos']
            ostime = ostime.astype(np.float32)
            info = np.concatenate(([age], tumorpos))
            info = info.astype(np.float32)

            shape = np.array(data.shape[1:])  # slice shape，(x, y, z)
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]),
                                 (pad_left[2], pad_right[2])))

            images = data[:4]

            # intensity normalization using z-score
            data_def = (data_def - data_def.mean()) / (data_def.std() + 1e-8)

            label = data[-1:]
            self.datas.append(images)
            self.datas_def.append(data_def)
            self.deaths.append(death)
            self.ostimes.append(ostime)
            self.info.append(info)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return {'data': self.datas[index], 'data_def': self.datas_def[index], 'death': self.deaths[index], 'ostime': self.ostimes[index], 'info': self.info[index]}


def get_dataset(fold, mode, input_size):
    # list data path and properties
    with open(os.path.join(config.DATASET.ROOT, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)[fold]
    datas = splits['train'] if mode == 'train' else splits['val']
    print(datas.shape)
    dataset = OrderedDict()
    for name in datas:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(config.DATASET.ROOT, name + '.npz')
        dataset[name]['path1'] = os.path.join(config.DATASET.DEF_ROOT, name + '.npz')
    assert len(config.MODEL.INPUT_SIZE) == 3, 'must be 3 dimensional patch size'
    return Dataset3D_itemread(dataset, config.MODEL.INPUT_SIZE, input_size)


def get_dataset_BraTS20(mode):
    # list data path and properties
    with open(os.path.join(config.DATASET.TEST_ROOT, 'splits_2020.pkl'), 'rb') as f:
        splits = pickle.load(f)
    datas = splits['train'] if mode == 'train' else splits['val']
    print(datas.shape)
    dataset = OrderedDict()
    for name in datas:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(config.DATASET.TEST_ROOT, name + '.npz')
        dataset[name]['path1'] = os.path.join(config.DATASET.TEST_DEF_ROOT, name + '.npz')
    assert len(config.MODEL.INPUT_SIZE) == 3, 'must be 3 dimensional patch size'
    return Dataset3D_itemread_BraTS20(dataset, config.MODEL.INPUT_SIZE)


def product_sum(row):
    return sum(row)
