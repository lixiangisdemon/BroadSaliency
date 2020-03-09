#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:43:45 2019

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
import random
import os
epochs = 400
save_dir = './weights'
root_dir = '/home/vision/lixiang/Dataset/MSRA-B'
batch_size = 256
csv_data = 'train.csv'

if (os.path.exists(save_dir) == False):
    os.mkdir(save_dir)

dataset = MyDataset(csv_data, root_dir + '/Img', root_dir + '/GT', 
                    Transforms=transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.06)
                    )
dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
Net = SalNet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Net.to(device)

#optimizer = torch.optim.SGD(Net.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12000, 20000], gamma=0.1)
optimizer = torch.optim.Adam(Net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps = 1e-8, weight_decay=1e-6)

def loss_function(act1, act2, act3, gt1, gt2, gt3):
    fbce = torch.nn.BCELoss(reduction='mean')
    fmse = torch.nn.MSELoss(reduction='mean')
    loss = 0.3 * fbce(act1, gt1) + 0.3 * fbce(act2, gt2) + 0.4 * fbce(act3, gt3)
    return loss

print ('starting to train')
for epoch in range(epochs):
    running_loss = 0.0
    for step, data in enumerate(dataloaders):
        imgs = data['image'].to(device)
        gts = data['gt'].to(device)
        gts160 = data['gt160'].to(device)
        feats = data['feature'].to(device)
        labels = data['label'].to(device)
        optimizer.zero_grad()
        act1, act2, act3 = Net({'images':imgs, 'features': feats})
        # print ("output1 : ", act1.shape, act1.max(), act1.min())
        # print ("output2 : ", act2.shape, act2.max(), act2.min())
        # print ("output3 : ", act3.shape, act3.max(), act3.min())
        # print ("gt : " , gts.shape, gts.max(), gts.min())
        # print ("labels : ", labels.shape, labels.max(), labels.min())
        loss = loss_function(act1, act2, act3, labels, gts160, gts)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (step % 10 == 0):
            print('[iteration: %5d] loss: %.3f'%(step, running_loss / 10))
            running_loss = 0.0
            torch.save(Net.state_dict(), '{}/weight{}_{}.pth'.format(save_dir, epoch, step))
        # scheduler.step()

print('finish training')
torch.save(Net.state_dict(), save_dir + '/final.pth')
