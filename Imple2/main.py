# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 23:57:56 2021

@author: QWERTY
"""
print('Importing Libraries')
print('########################################################################################################################')
# Importing libraries
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

import zipfile

# Loading Dataset file

from Load_Dataset import Train_Set,Validation_Set

# Loading Model class

from Model_class.MLP_5layer_1024_grad_red_units_geluActivation_Concept5 import MLP


######################################################################################################################################
######################################################HYPERPARAMETER DEFINITIONS######################################################
#####################################################################################################################################

# Context length (Defines size of frame for better prediction) (Also helps in building up model by providing input size)
Context_length = 5
# Batch Size
train_batch_size = 128
test_batch_size = 128

# Epoch Size
epochs = 5
# Learning Rate
lr = 0.0001



# Device loading
#Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################################################################################
##############################################LOADING DATA FILES######################################################
#####################################################################################################################################
print('LOADING DATA')
print('########################################################################################################################')

# Loading Training set and activating dataloader
train_set = Train_Set(Context_length=Context_length)

train_loader = DataLoader(train_set,batch_size=train_batch_size,shuffle=True,num_workers=4)

# Loading Validation set and activating data loaser
validation_set = Validation_Set(Context_length=Context_length)

validation_loader = DataLoader(validation_set,batch_size=test_batch_size,shuffle=True,num_workers=4)

######################################################################################################################################
######################################LOADING MODEL AND SETTING UP TRAINING PARAMETERS######################################################
#####################################################################################################################################
print('LOADING MODEL')
print('########################################################################################################################')

# Defining Model 
net = MLP(Context_length).to(device)

# Defining loss function
criterion = nn.CrossEntropyLoss()

#Optimizer selection
#optimizer = optim.SGD(net.parameters(),lr = 0.0001)
optimizer = optim.Adam(net.parameters())

######################################################################################################################################
######################################WRITER DESCRIPTION######################################################
#####################################################################################################################################


# Defining summary writer to save model parameters and graph
PATH = '/scratch/shanbhag.d/Phoneme_Recognition/log/MLP_5layer_1024_grad_red_units_geluActivation_Concept5' 
writer = SummaryWriter(PATH+'/writer')

######################################################################################################################################
##################################TO RELOAD MODEL AND PARAMETERS FOR CONTINUINING TRAINING######################################################
#####################################################################################################################################

# Defining parameters to continue training stopped due to whatever reason
try:
    checkpoint = torch.load(PATH+'/saved_model/model_data',map_location=device)
    
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch'] + 1

except:
    epoch_start = 0
  
######################################################################################################################################
#########################################TRAINING LOOP######################################################
#####################################################################################################################################
print('TRAINING')
print('########################################################################################################################')
# Starting training loop
for epoch in range(epoch_start,epochs):

    # Initialize losses
    running_loss_per_epoch = 0
    running_accuracy_per_epoch = 0
    count = 0
    
    # Iterating over whole datset once
    for i,data in enumerate(train_loader):
    
        # Specify that we are in training loop
        net.train()
      
        # Fetching features and lables from loaded data and loading onto device
        features = data['Feature_Vector'].to(device)
        labels = data['Feature_Label'].to(device)
      
        # Lets remove the accumulated gradient 
        optimizer.zero_grad()
      
        # Forward + Bachward +gradient accumulation
        outputs = torch.squeeze(net(features))
      
        
        # Calculating loss
        loss = criterion(outputs,labels.long())
        # Backward pass
        loss.backward()
        # update Weights
        optimizer.step()
      
        # Calculating Accuracy
        acc = torch.sum(torch.argmax(outputs,dim=1)==labels).float()/np.shape(labels)[0]
      
        # Updating Statistics
        running_loss_per_epoch += loss.item()
        running_accuracy_per_epoch += acc.item()
        count += 1
      
        # Validation loop every 2000 iterations
        if i%2000 ==0:
            # Printing training stats
            print('------------------------------------------------------------------------------------------------------')
            print('Epoch:',epoch+1)
            print('Iteration: %d Average Training Loss: %.3f Average Training Accuracy: %.3f' % (i, running_loss_per_epoch / count, running_accuracy_per_epoch / count))
            writer.add_scalar('Average Training Loss : ',running_loss_per_epoch / count,epoch * len(train_loader) + i)
            writer.add_scalar('Average Training Accuracy : ',running_accuracy_per_epoch / count,epoch * len(train_loader) + i)
            # Evaluation mode
            net.eval()
            # Initialize Validation stats
            running_val_loss = 0
            running_val_acc  = 0
            val_count = 0
        
            # Reinitialize Training Stats
            running_loss_per_epoch = 0
            running_accuracy_per_epoch = 0
            count = 0
        
            with torch.no_grad():
                for val_i,val_data in enumerate(validation_loader):
                    val_feature, val_labels = val_data['Feature_Vector'].to(device),val_data['Feature_Label'].to(device)
                    val_outputs = torch.squeeze(net(val_feature))
                    running_val_loss += criterion(val_outputs, val_labels.long()).item()
                    running_val_acc += (torch.sum(torch.argmax(val_outputs,dim=1) == val_labels).float()/np.shape(val_labels)[0]).item()
                    val_count += 1
                    if (val_i+1)%2000 == 0:
                        break
                    
            print('Average Validation Loss:',running_val_loss/val_count,' Average Validation Accuracy:',running_val_acc/val_count)
            writer.add_scalar('Average Validation Loss',running_val_loss / val_count,epoch * len(train_loader) + i)
            writer.add_scalar('Average Validation Accuracy',running_val_acc / val_count,epoch * len(train_loader) + i)
        
    # Saving Model
    torch.save({'epoch': epoch,'model_state_dict': net.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, PATH+'/saved_model/model_data')
      
