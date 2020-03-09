#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:35:20 2019

@author: lixiang
"""
import torch
import torchvision

class SalNet(torch.nn.Module):
    def __init__(self, input_size = 32, input_dim = 72, phase='train'):
        super(SalNet, self).__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.input_dim = input_dim
        
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=5, stride=2), torch.nn.LeakyReLU(0.05),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, dilation = 2), torch.nn.LeakyReLU(0.05),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, dilation = 2), torch.nn.LeakyReLU(0.05),
            torch.nn.MaxPool2d(kernel_size=3, stride=1)
        )

        self.norm0 = torch.nn.BatchNorm2d(32)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_dim, 64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(64, 64, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(0.05)
            )

        self.norm1 = torch.nn.BatchNorm2d(128) 

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05), 
            torch.nn.Conv2d(128, 128, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(0.05)
            )
        
        self.norm2 = torch.nn.BatchNorm2d(256)

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1), torch.nn.LeakyReLU(0.05)
            )
        
        self.norm3 = torch.nn.BatchNorm2d(512)

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=0, output_padding=0), 
            torch.nn.LeakyReLU(0.05)
            )
        
        self.norm4 = torch.nn.BatchNorm2d(256)
        
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            torch.nn.LeakyReLU(0.05)
            )
        
        self.norm5 = torch.nn.BatchNorm2d(128)
        
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(128, 128, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            torch.nn.LeakyReLU(0.05)
            )
        self.conv6_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
            )
        
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(64 + 32, 64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(64, 64, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(32, 32, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(16, 16, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            )
        self.conv7_1 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
            )
        
        self.conv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1, output_padding=0), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(16, 16, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1), torch.nn.LeakyReLU(0.05),
            torch.nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
            )
        self.act = torch.nn.Sigmoid()
        
    def forward(self, input):
        
        imgs = input['images']
        feats = input['features']
        out0 = self.norm0(self.conv0(imgs))
        out1 = self.norm1(self.conv1(feats))
        out2 = self.norm2(self.conv2(out1))
        out3 = self.norm3(self.conv3(out2))
        out4 = self.norm4(self.conv4(out3))
        out4_2 = torch.cat((out2, out4), dim=1)
        out5 = self.norm5(self.conv5(out4_2))
        out5_1 = torch.cat((out5, out1), dim=1)
        out6 = self.conv6(out5_1)
        act1 = self.act(self.conv6_1(out6))
        out7 = self.conv7(torch.cat((out6, out0), dim=1) * (act1 + 0.1))
        act2 = self.act(self.conv7_1(out7))
        out8 = self.conv8(out7 * (act2 + 0.1))
        act3 = self.act(out8)
        return act1, act2, act3
    
class BLS(torch.nn.Module):
    def __init__(self, input_dims, output_dims, feature_dims = 64, enhancement_dims = 1024):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.feature_dims = feature_dims
        self.enhancement_dims = enhancement_dims
        self.We = torch.rand((self.input_dims, self.feature_dims), dtype=torch.float32, requires_grad=False) * 0.1 - 0.05
        self.be = torch.ones((1, self.feature_dims), dtype=torch.float32, requires_grad=False) * 0.01
        self.Wh = torch.rand((self.feature_dims, self.enhancement_dims), dtype=torch.float32, requires_grad=False) * 0.1 - 0.05
        self.bh = torch.ones((1, self.enhancement_dims), dtype=torch.float32, requires_grad=False) * 0.01
        self.tanh = torch.nn.Tanh()
        self.linear = torch.nn.Linear(self.feature_dims + self.enhancement_dims, self.output_dims, bias = True)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input):
        input = input.squeeze(0)
        features = torch.matmul(input, self.We) + self.be
        enhancements = torch.matmul(self.tanh(features), self.Wh) + self.bn
        Z = torch.cat((features, enhancements), dim=1)
        output = self.sigmoid(self.linear(Z))
        return output
            
            
            
            
