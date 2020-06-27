#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:03:37 2020

@author: minseongkang
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import torch.nn.functional as func

#processing the dataset
train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train,batch_size=100,shuffle=True)
test_loader = torch.utils.data.DataLoader(test,batch_size=100,shuffle=False)

#initializing weights
W = torch.randn(784,10)/np.sqrt(784)
W.requires_grad_()

#bias
b = torch.zeros(10, requires_grad=True)

#optimizer
opt = torch.optim.SGD([W,b],lr=0.2)

#training for each batch
for images, labels in tqdm(train_loader):
    opt.zero_grad()
    x = images.view(-1,28*28)
    y = torch.matmul(x,W)+b
    ce = func.cross_entropy(y, labels)
    ce.backward()
    opt.step()

#evaluation
correct = 0 
total = len(test)

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        x = images.view(-1,28*28)
        y = torch.matmul(x,W)+b
        
        preds = torch.argmax(y,dim=1)
        correct += torch.sum((preds==labels).float())

print('Test accuracy: {}'.format(correct/total))