# -*- coding: utf-8 -*-
"""
@Title :  Training Dataset class

@author: Dwijay
"""
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


######################################################################################################################################
######################################################TRAINING CLASS######################################################
#####################################################################################################################################



# Defining Training Dataset Class
class Train_Set(Dataset):

    def __init__(self,Context_length):
        '''
        Initialization function
        
        INPUT:
          train_feature : Contains train feature values
          train_label : Contains train labels
          Concept_length : This variable mentions the number of frames to concatenate on both the sides of focus frame
        
        OUTPUT:
          None
        '''
        # Assigning data variables
        self.train_data = np.load('/scratch/shanbhag.d/Phoneme_Recognition/Unzip_Dataset/train.npy',allow_pickle=True)
        self.train_label= np.load('/scratch/shanbhag.d/Phoneme_Recognition/Unzip_Dataset/train_labels.npy',allow_pickle=True)
        # Defining Concept space
        self.k = Context_length
        
        # Constructing a list to contain the total number of frames seen so far
        # Each element in this list is number of frames in present sample + number of frames seen so far
        # Initializing list
        self.Frame_list = []
        # Initializing the variable to store total number of frames in dataset
        self.total_length = 0
        # Iterating through the whole data samples to fetch and accumulate the frame numbers
        for i in self.train_label:
          # Calculating numbers of frames in present sample and adding it to the total length variable
          self.total_length += np.shape(i)[0]
          # appending this to the Frame_list
          self.Frame_list.append(self.total_length)
        
        # Defining size of 1 frame
        self.frame_size = 40


    def __len__(self):
        '''
        Function to return len of total dataset
        '''
        #Return the length
        return self.total_length
      
    def __getitem__(self,idx):
        '''
        Function to fetch data sample using idx
        '''
        # Fetch sample id from the given idx value
        sample_id = self.Fetch_Frame_info(idx)
        # Fetch frame is from the sample id found
        if sample_id != 0:
          frame_id = idx - self.Frame_list[sample_id-1]
        else:
          frame_id = idx
      
        # fetching feature vector and feature label
        feature_vector = torch.from_numpy(np.asarray(self.concat_context_frame(sample_id,frame_id))).float()
        feature_label = torch.from_numpy(np.asarray(self.train_label[sample_id][frame_id]))
      
        # return the dictionary
        return {'Feature_Vector':feature_vector,'Feature_Label':feature_label}
      
    
    def concat_context_frame(self,sample_id,frame_id):
        '''
        Return The frame with context
      
        INPUT:
          sample_id : sample number 
          frame_id : Frame number
        RETURN:
          Frame with context
        '''
        # Pick up the targeted sample
        sample = self.train_data[sample_id]
      
        # Fetch info about padding
        pad_info = self.Check_zero_pad(sample_id,frame_id)
      
        # If we need to pad
        if (pad_info['do_we_need_zero_pad']==1):
      
          # Calculate the number of elements in one frame with context
          frame_ele_num = ((2*self.k)+1) * self.frame_size
          # Define feature vector as zeros
          feature_vector = np.zeros((1,frame_ele_num))
      
          # Check if we need left padding
          if (pad_info['left_zero_pad']==1) and (pad_info['right_zero_pad']==0):
      
            # Fill in the frame information
            feature_vector[0,pad_info['no_frame_left']*self.frame_size:] = np.reshape(sample[:frame_id+1+self.k,:],(1,-1))
      
          # Check if we need right padding
          if (pad_info['left_zero_pad']==0) and (pad_info['right_zero_pad']==1):
      
            # Variable to update frames to add
            to_add = ((2*self.k)+1) - pad_info['no_frame_right']
      
            # Fill in the frame information
            feature_vector[0,:to_add*self.frame_size] = np.reshape(sample[frame_id-self.k:,:],(1,-1))
      
          # Check if we need to pad on left and right
          if (pad_info['left_zero_pad']==1) and (pad_info['right_zero_pad']==1):
      
            # Variable to update frames to add
            to_add = ((2*self.k)+1) - pad_info['no_frame_right']
      
            # Fill in the frame information
            feature_vector[0,pad_info['no_frame_left']*self.frame_size:to_add*self.frame_size] = np.reshape(sample[:,:],(1,-1))
      
      
        else:
          # If we dont need to pad
          feature_vector = np.reshape(sample[(frame_id-self.k):(frame_id+self.k+1),:],(1,-1))
      
        
        # Return feature Vector
        return feature_vector
      
    
    
    
    
    def Check_zero_pad(self,sample_id,frame_id):
    
        '''
        Return if we need padding and on which side
      
        INPUT:
          sample_id : sample number 
          frame_id : Frame number
        RETURN:
          Info about padding zeros 
        '''
        # Initializing variables
        do_we_need_zero_pad = 0
        left_zero_pad = 0
        right_zero_pad = 0
        no_frame_left = 0
        no_frame_right = 0
      
        # Check if we need to add zeros to left of frame
        if frame_id - self.k < 0:
          do_we_need_zero_pad = 1
          left_zero_pad = 1
          no_frame_left = self.k - frame_id
      
        # Find total size of sample
        if sample_id != 0:
          tot_size = self.Frame_list[sample_id]-self.Frame_list[sample_id-1]
        else:
          tot_size = self.Frame_list[sample_id]
      
        # Check if we need to add zeros to the right of frame
        if frame_id + self.k >= tot_size:
          do_we_need_zero_pad = 1
          right_zero_pad = 1
          no_frame_right = frame_id + self.k - tot_size +1
      
        # Returning the info dictionary
        return {'do_we_need_zero_pad':do_we_need_zero_pad,'left_zero_pad':left_zero_pad,'right_zero_pad':right_zero_pad,'no_frame_left':no_frame_left,'no_frame_right':no_frame_right}
      
      
    
      
    
    def Fetch_Frame_info(self,idx):
        '''
          Function to fetch the current sample to which the frame belongs to.
      
          This function can be made more efficient by checking if the idx is more than half or less than half of total size
        '''
        # Iterating through the Frame list
        for i in range(len(self.Frame_list)):
          # Checking if idx is less than  number of frames seen
          if self.Frame_list[i]>idx:
            return i
        
        
        
######################################################################################################################################
######################################################VALIDATION CLASS######################################################
#####################################################################################################################################

        
        
class Validation_Set(Dataset):

    def __init__(self,Context_length):
        '''
        Initialization function
      
        INPUT:
          train_feature : Contains train feature values
          train_label : Contains train labels
          Concept_length : This variable mentions the number of frames to concatenate on both the sides of focus frame
      
        OUTPUT:
          None
        '''
        
        # Assigning data variables
        self.train_data = np.load('/scratch/shanbhag.d/Phoneme_Recognition/Unzip_Dataset/dev.npy',allow_pickle=True)
        self.train_label= np.load('/scratch/shanbhag.d/Phoneme_Recognition/Unzip_Dataset/dev_labels.npy',allow_pickle=True)
        # Defining Concept space
        self.k = Context_length
      
        # Constructing a list to contain the total number of frames seen so far
        # Each element in this list is number of frames in present sample + number of frames seen so far
        # Initializing list
        self.Frame_list = []
        # Initializing the variable to store total number of frames in dataset
        self.total_length = 0
        # Iterating through the whole data samples to fetch and accumulate the frame numbers
        for i in self.train_label:
          # Calculating numbers of frames in present sample and adding it to the total length variable
          self.total_length += np.shape(i)[0]
          # appending this to the Frame_list
          self.Frame_list.append(self.total_length)
      
        # Defining size of 1 frame
        self.frame_size = 40
    
    
    def __len__(self):
        '''
        Function to return len of total dataset
        '''
        #Return the length
        return self.total_length
      
    def __getitem__(self,idx):
        '''
        Function to fetch data sample using idx
        '''
        # Fetch sample id from the given idx value
        sample_id = self.Fetch_Frame_info(idx)
        # Fetch frame is from the sample id found
        if sample_id != 0:
          frame_id = idx - self.Frame_list[sample_id-1]
        else:
          frame_id = idx
      
        # fetching feature vector and feature label
        feature_vector = torch.from_numpy(np.asarray(self.concat_context_frame(sample_id,frame_id))).float()
        feature_label = torch.from_numpy(np.asarray(self.train_label[sample_id][frame_id]))
      
        # return the dictionary
        return {'Feature_Vector':feature_vector,'Feature_Label':feature_label}
      
    
    def concat_context_frame(self,sample_id,frame_id):
        '''
        Return The frame with context
      
        INPUT:
          sample_id : sample number 
          frame_id : Frame number
        RETURN:
          Frame with context
        '''
        # Pick up the targeted sample
        sample = self.train_data[sample_id]
      
        # Fetch info about padding
        pad_info = self.Check_zero_pad(sample_id,frame_id)
      
        # If we need to pad
        if (pad_info['do_we_need_zero_pad']==1):
      
          # Calculate the number of elements in one frame with context
          frame_ele_num = ((2*self.k)+1) * self.frame_size
          # Define feature vector as zeros
          feature_vector = np.zeros((1,frame_ele_num))
      
          # Check if we need left padding
          if (pad_info['left_zero_pad']==1) and (pad_info['right_zero_pad']==0):
      
            # Fill in the frame information
            feature_vector[0,pad_info['no_frame_left']*self.frame_size:] = np.reshape(sample[:frame_id+1+self.k,:],(1,-1))
      
          # Check if we need right padding
          if (pad_info['left_zero_pad']==0) and (pad_info['right_zero_pad']==1):
      
            # Variable to update frames to add
            to_add = ((2*self.k)+1) - pad_info['no_frame_right']
      
            # Fill in the frame information
            feature_vector[0,:to_add*self.frame_size] = np.reshape(sample[frame_id-self.k:,:],(1,-1))
      
          # Check if we need to pad on left and right
          if (pad_info['left_zero_pad']==1) and (pad_info['right_zero_pad']==1):
      
            # Variable to update frames to add
            to_add = ((2*self.k)+1) - pad_info['no_frame_right']
      
            # Fill in the frame information
            feature_vector[0,pad_info['no_frame_left']*self.frame_size:to_add*self.frame_size] = np.reshape(sample[:,:],(1,-1))
      
      
        else:
          # If we dont need to pad
          feature_vector = np.reshape(sample[(frame_id-self.k):(frame_id+self.k+1),:],(1,-1))
      
        
        # Return feature Vector
        return feature_vector
      
    
    
    
    
    def Check_zero_pad(self,sample_id,frame_id):
        '''
        Return if we need padding and on which side
      
        INPUT:
          sample_id : sample number 
          frame_id : Frame number
        RETURN:
          Info about padding zeros 
        '''
        # Initializing variables
        do_we_need_zero_pad = 0
        left_zero_pad = 0
        right_zero_pad = 0
        no_frame_left = 0
        no_frame_right = 0
      
        # Check if we need to add zeros to left of frame
        if frame_id - self.k < 0:
          do_we_need_zero_pad = 1
          left_zero_pad = 1
          no_frame_left = self.k - frame_id
      
        # Find total size of sample
        if sample_id != 0:
          tot_size = self.Frame_list[sample_id]-self.Frame_list[sample_id-1]
        else:
          tot_size = self.Frame_list[sample_id]
      
        # Check if we need to add zeros to the right of frame
        if frame_id + self.k >= tot_size:
          do_we_need_zero_pad = 1
          right_zero_pad = 1
          no_frame_right = frame_id + self.k - tot_size +1
      
        # Returning the info dictionary
        return {'do_we_need_zero_pad':do_we_need_zero_pad,'left_zero_pad':left_zero_pad,'right_zero_pad':right_zero_pad,'no_frame_left':no_frame_left,'no_frame_right':no_frame_right}
      
      
    
      
    
    def Fetch_Frame_info(self,idx):
        '''
          Function to fetch the current sample to which the frame belongs to.
      
          This function can be made more efficient by checking if the idx is more than half or less than half of total size
        '''
        # Iterating through the Frame list
        for i in range(len(self.Frame_list)):
          # Checking if idx is less than  number of frames seen
          if self.Frame_list[i]>idx:
            return i



