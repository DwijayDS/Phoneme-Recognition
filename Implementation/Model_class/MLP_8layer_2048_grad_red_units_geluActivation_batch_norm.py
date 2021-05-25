# -*- coding: utf-8 -*-
"""
@title: Model Definition (5 Hidden Layer, Size (2048,1024,512,512,256,128) gelu activation )

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
      
        # Difining hidden layer size
        hid1 = 2048
      
        hid2 = 2048
      
        hid3 = 1024
      
        hid4 = 1024
      
        hid5 = 512
        
        hid6 = 512
        
        hid7 = 256
        
        hid8 = 256
        
        hid9 = 128
        
        hid10 = 128
      
        # Difining layers
        self.layer1 = nn.Linear(self.inp_vec_len,hid1)
      
        self.layer2 = nn.Linear(hid1,hid2)
        
        self.dense1_bn = nn.BatchNorm1d(hid2)
      
        self.layer3 = nn.Linear(hid2,hid3)
      
        self.layer4 = nn.Linear(hid3,hid4)
        
        self.dense2_bn = nn.BatchNorm1d(hid4)
      
        self.layer5 = nn.Linear(hid4,hid5)
        
        self.layer6 = nn.Linear(hid5,hid6)
        
        self.dense3_bn = nn.BatchNorm1d(hid6)
        
        self.layer7 = nn.Linear(hid6,hid7)
        
        self.layer8 = nn.Linear(hid7,hid8)

        self.dense4_bn = nn.BatchNorm1d(hid8)
        
        self.layer9 = nn.Linear(hid8,hid9)
        
        self.layer10 = nn.Linear(hid9,hid10)
              
        self.layer11 = nn.Linear(hid10,self.n_classes)
      
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
        x = F.gelu(self.layer1(x))
      
        x = F.gelu(self.dense1_bn(self.layer2(x)))
      
        x = F.gelu(self.layer3(x))
        
        # Relu starts from here
      
        x = F.gelu(self.dense2_bn(self.layer4(x)))
      
        x = F.gelu(self.layer5(x))
        
        x = F.gelu(self.dense3_bn(self.layer6(x)))
        
        x = F.gelu(self.layer7(x))
        
        x = F.gelu(self.dense4_bn(self.layer8(x)))

        x = F.gelu(self.layer9(x))
        
        x = F.gelu(self.layer10(x))
              
        x = self.layer11(x)   # We dont use activation here Because we are using cross entropy loss which has inbuild log softmax operation
      
        return x


