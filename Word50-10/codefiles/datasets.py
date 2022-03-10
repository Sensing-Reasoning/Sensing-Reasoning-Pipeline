from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from typing import *
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
import json
import numpy as np
from PIL import Image
from scipy.io import loadmat 

# list of all datasets
DATASETS = ["word10_main","word10_character"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "word10_main":
        return _word10_main(split)
    elif dataset == "word10_character":
        return _word10_character(split)
    

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "word10_main":
        return 10
    elif dataset == "word10_character":
        return 26

def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    elif dataset == "word10_main":
        return NormalizeLayer(_WORD10_MEAN, _WORD10_STDDEV)
    elif dataset == "word10_character":
        return NormalizeLayer(_WORD10_MEAN, _WORD10_STDDEV)

_WORD10_MEAN = [0.5,]
_WORD10_STDDEV = [0.5,]


def _word10_character(split: str) -> Dataset:
    tar_dir = "../data/word50.mat"
    word50 = loadmat(tar_dir)
    if split == "train":
        data = word50['train_feat']
        data = torch.FloatTensor(data).T.reshape(-1,1,28,28)
        label = word50['train_label']
        label = torch.LongTensor(label).squeeze()
    elif split == "test":
        data = word50['test_feat']
        data = torch.FloatTensor(data).T.reshape(-1,1,28,28)
        label = word50['test_label']
        label = torch.LongTensor(label).squeeze()
    return Word10CharacterFolder(data, label)

def _word10_main(split: str) -> Dataset:
    tar_dir = "../data/word10.pt"
    word10 = torch.load(tar_dir)
    if split == "train":
        data = word10['train_all_feat']
        label = word10['train_all_label']
    elif split == "test":
        data = word10['test_all_feat']
        label = word10['test_all_label']
    return Word10MainFolder(data, label)

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        if len(input.shape) == 2:
            return (input - self.means) / self.sds
        else:
            (batch_size, num_channels, height, width) = input.shape
            means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
            sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
        
class Word10CharacterFolder(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x = self.data[index]  
        y = self.label[index] 
        return x,y 

    def samples_weights(self):
        labels = pd.value_counts(self.label.numpy()).sort_index()
        weights = 1./ torch.tensor(labels.to_list(), dtype=torch.float)
        tmp = torch.zeros(26)
        tmp[:23] = weights[:23]
        tmp[-2:] = weights[-2:]
        weights = tmp
        samples_weights = weights[self.label]
        return samples_weights
    
    def __len__(self):
        return self.data.shape[0] 
    
class Word10MainFolder(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x = self.data[index]  
        y = self.label[index] 
        return x,y 

    def samples_weights(self):
        labels = pd.value_counts(self.label.numpy()).sort_index()
        weights = 1./ torch.tensor(labels.to_list(), dtype=torch.float)
        samples_weights = weights[self.label]
        return samples_weights
    
    def __len__(self):
        return self.data.shape[0] 