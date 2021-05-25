# -*- coding: utf-8 -*-
"""
@title: Model Definition (CNN 5 layers deep  gelu activation )

@author: QWERTY
"""
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from scipy.stats import mode


class MLP(nn.Module):

    def __init__(self,Context_length):
        # Fetching __init__
        super(MLP,self).__init__()
      
        # Setting context frame length 
        self.k = Context_length
      
        # Finding the input vector length
        self.inp_vec_len = 40*((2*self.k)+1)
      
        # output vector size
        self.n_classes = 71
      
        # Filter size
        ker_size = 3
        # Difining hidden layer size
        feat1 = 2
        
        feat2 = 4
        
        feat3 = 8
        
        feat4 = 16
        
        feat5 = 32
        
        feat6 = 32
        
        hid1 = 1024
      
        hid2 = 512
      
        hid3 = 256
      
        hid4 = 128
      
        hid5 = 128
        
        hid6 = 90
        
        
        
      
        # Difining layers
        self.conv1 = nn.Conv1d(1,feat1,3,2)
        self.conv2 = nn.Conv1d(feat1,feat2,3,2)
        self.conv3 = nn.Conv1d(feat2,feat3,3,2)
        self.conv4 = nn.Conv1d(feat3,feat4,3,2)
        self.conv5 = nn.Conv1d(feat4,feat5,3,2)
        self.conv6 = nn.Conv1d(feat5,feat6,3,2)
        
        self.layer1 = nn.Linear(13*feat6,hid1)
      
        self.layer2 = nn.Linear(hid1,hid2)
        
        self.dense1_bn = nn.BatchNorm1d(hid2)
      
        self.layer3 = nn.Linear(hid2,hid3)
      
        self.layer4 = nn.Linear(hid3,hid4)
        
        self.dense2_bn = nn.BatchNorm1d(hid4)
      
        self.layer5 = nn.Linear(hid4,hid5)
        
        self.layer6 = nn.Linear(hid5,hid6)
        
        self.layer7 = nn.Linear(hid6,self.n_classes)
        
        
        
        
        
        
        
    def forward(self,x):
        '''
          Writing code for forward pass.
      
          We start with relu activation, but this must be updated since the input is positive and negative and by using relu in the initial layers we reduce the effect of
          negative features.
          Check leaky relu or GELU or other activation functions.
          If our data variance is important we can also add BatchNormalization
          
          Jacob Lee :
      
          "The values present in your data also inform what you choose. ReLU zeros out negative values but leaves positive ones so it may not be a very 
          informative activation near the front of a network if there are only positive values in the input.
          If theres significant variance in your input data, batchnorm is commonly used.
          And if you have data where each observation is interrelated, CNNs are used for short term and RNNs are used for long term relationships. 
          There are theoretical justifications, but still no reliable algorithm for determining architecture choice. Hard problem; massive search space
          "
        '''
        # Softmax Function turns large negative numbers to zero probability and small negative numbers to small probability. Gelu for large negative number is anyhow zero 
        # But we dont know the weight of 0.1 vs -0.1. And with softmax, softmax(-0.1)<softmax(0.1). Therefore, lets have Relu in last two hidden layers
        x = torch.unsqueeze(x,1)
        
        x = F.gelu(self.conv1(x))
        
        x = F.gelu(self.conv2(x))
        
        x = F.gelu(self.conv3(x))
        
        x = F.gelu(self.conv4(x))
        
        x = F.gelu(self.conv5(x))
        
        x = F.gelu(self.conv6(x))
        
        x = torch.flatten(x,1)
        
        x = F.gelu(self.layer1(x))
      
        x = F.gelu(self.dense1_bn(self.layer2(x)))
      
        x = F.gelu(self.layer3(x))
        
        # Relu starts from here
      
        x = F.gelu(self.dense2_bn(self.layer4(x)))
      
        x = F.gelu(self.layer5(x))
        
        x = F.gelu(self.layer6(x))
        
        
        x = self.layer7(x)   # We dont use activation here Because we are using cross entropy loss which has inbuild log softmax operation
      
        return x


