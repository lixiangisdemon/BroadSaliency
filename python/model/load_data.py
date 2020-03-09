#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:43:24 2019

@author: lixiang
"""
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import random
import cv2 as cv
import os
from utils.extrafeature import ExtraFeature

def img_show(img, img_name = 'tmp'):
    plt.imshow(img)
    plt.title(img_name)

class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, gt_dir, feat_size = 40, Transforms=None, train = True, shuffle = False):
        self.img_dir = img_dir.rstrip('/') + '/'
        self.gt_dir = gt_dir.rstrip('/') + '/'
        self.transforms = Transforms
        self.train = train
        self.feat_size = feat_size
        self.E = ExtraFeature(feat_size=40)
        
        if (csv_file is not None) and (os.path.exists(csv_file)):
            self.csv = pd.read_csv(csv_file)
            self.names = list(np.array(self.csv['name']))
            self.lens = len(self.names)
        else:
            self.names = [name for name in os.listdir(img_dir) if name.endswith('jpg') or name.endswith('png')]
            self.lens = len(self.names)
            
    def get_name(self, idx):
        return self.names[idx]
        
    def __len__(self):
        return self.lens
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, list):
            idx = [id for id in idx]
        else:
            idx = idx
        name = self.names[idx]
        return self._get_sample(name)
    
    def _get_sample(self, name):
        image = Image.open(self.img_dir + name)
        if (image.mode == 'RGBA'):
            image = image.convert('RGB')
        w, h = image.size
        gt = None
        if self.train:
            gt = np.array(Image.open(self.gt_dir + name[:-3] + 'png'), dtype=np.float32) / 255.0
        else:
            gt = np.zeros((h, w), dtype=np.float32)
            
        if (self.transforms is not None):
            image = self.transforms(image)
            
        img = np.array(image)
        img = cv.resize(img, (400,400))
        origin_shape = gt.shape
        gt = cv.resize(gt, (400,400))
        [_, res1, res2] = self.E.extract_feature(img, gt)
        [features, label] = self.E.cvt_gridfeature(res1, res2)
        features = features.transpose((1, 0))
        label = np.reshape(label, (1, -1))

        gt = cv.resize(gt, (320, 320))
        gt160 = cv.resize(gt, (160, 160))
        features = features.reshape((-1, self.feat_size, self.feat_size))
        label = label.reshape((-1, self.feat_size, self.feat_size))
        #label = cv.resize(gt, (32, 32)).transpose(2,0,1)[:1, :, :]
        sample = {'image': torch.from_numpy(np.array(img.transpose(2,0,1), dtype=np.float32)),
                  'gt_shape': origin_shape,
                  'gt': torch.from_numpy(np.array(gt[np.newaxis, :, :], dtype=np.float32)),
                  'gt160': torch.from_numpy(np.array(gt160[np.newaxis, :, :], dtype=np.float32)),
                  'feature': torch.from_numpy(np.array(features, dtype=np.float32)),
                    'label': torch.from_numpy(np.array(label, dtype=np.float32))}
        return sample
