#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:49:51 2019

@author: lixiang
"""
import sys
sys.path.append('../libs')
sys.path.append('./utils')
sys.path.append('./model')

import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from model.load_data import MyDataset
from model.Model import SalNet
import os
import cv2 as cv

weight = 'weights/final.pth'
root_dir = '/home/vision/lixiang/Dataset/ECSSD'
batch_size = 1
csv_data = 'ecssd.csv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = SalNet()
if (os.path.exists(weight)):
    net.load_state_dict(torch.load(weight, map_location=torch.device(device)))
net.to(device)

dataset = MyDataset(csv_data, root_dir + '/Img', root_dir + 'GT', Transforms=None, train=False)
dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
MAE = 0.0
count = 0
for i, data in enumerate(dataloaders):
    imgs = data['image'].to(device)
    feats = data['feature'].to(device)
    gt = data['gt'].cpu().detach().numpy()[0,0,:,:]
    _, _, out = net({'images':imgs, 'features': feats})
    output = out.cpu().detach().numpy()[0,0,:,:]
    gt = cv.resize(gt, output.shape[::-1])
    mae = np.mean(np.abs(output - gt))
    MAE += mae
    count += 1
    
print ('final MAE is: {}'.format(MAE / count))
