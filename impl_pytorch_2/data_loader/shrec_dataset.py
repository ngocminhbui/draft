import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ArrayDataset(Dataset):
    """Constructs a simple dataset from array, matrix"""
    def __init__(self, x,y):
        self.X = x
        self.Y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

class ShrecMultiviewDataset(Dataset):
    """ShrecMultiviewDataset dataset."""
    ''' for sh '''
    def __init__(self, data_dir, train=True, transform=None):
        """
        Args:
            
        """
        self.data_dir = data_dir
        if train:
            self.X = np.load(os.path.join(self.data_dir, 'x_trainval_sample.npy'))
            self.Y = np.load(os.path.join(self.data_dir, 'y_trainval_sample.npy'))
        else:
            self.X = np.load(os.path.join(self.data_dir, 'x_test_sample.npy'))
            self.Y = np.load(os.path.join(self.data_dir, 'y_test_sample.npy'))
            
        #convert float64 to float32, label to long & squeezed
        self.X = np.float32(self.X)
        self.Y = np.int64(self.Y).squeeze()
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(sample)
            
        return (self.X[idx], self.Y[idx])