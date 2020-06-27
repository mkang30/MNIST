#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:56:12 2020

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

#initializing layer 1 weights and bias
W = torch.randn(784,500)/np.sqrt(784)
W.requires_grad_()
b = torch.zeros(500, requires_grad=True)

#initializing layer 2 weights and bias
V = torch.randn(500,10)/np.sqrt(500)
V.requires_grad_()
b2 = torch.zeros(10, requires_grad=True)

#optimizer
opt = torch.optim.SGD([W,b,V,b2],lr=1.0)

#training for each batch - 2 epochs
for i in range(3):
    for images, labels in tqdm(train_loader):
        opt.zero_grad()
        x = images.view(-1,28*28)
        #first layer
        a = func.relu(torch.matmul(x,W)+b)
        #second layer
        y = torch.matmul(a,V)+b2
        ce = func.cross_entropy(y, labels)
        ce.backward()
        opt.step()

correct = 0 
total = len(test)

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        x = images.view(-1,28*28)
        a = func.relu(torch.matmul(x,W)+b)
        y = torch.matmul(a,V)+b2
        preds = torch.argmax(y,dim=1)
        correct += torch.sum((preds==labels).float())

print('Test accuracy: {}'.format(correct/total))
