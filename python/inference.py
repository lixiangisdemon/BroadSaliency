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
root_dir = '.'
batch_size = 8
csv_data = 'train.csv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = SalNet()
if (os.path.exists(weight)):
    net.load_state_dict(torch.load(weight, map_location=torch.device(device)))
net.to(device)

img_name = '../fig/9.png'
dataset = MyDataset(csv_data, root_dir, root_dir, Transforms=None, train = False)
data = dataset._get_sample(img_name)
imgs = data['image'].to(device).unsqueeze(0)
feats = data['feature'].to(device).unsqueeze(0)
_, _, out = net({'images':imgs, 'features': feats})
output = out.cpu().detach().numpy()[0,0,:,:]
shape = data['gt_shape']
res = cv.resize(output, (shape[1], shape[0]))
plt.imshow(res)
plt.show()
cv.imwrite('9.png', res * 255)
