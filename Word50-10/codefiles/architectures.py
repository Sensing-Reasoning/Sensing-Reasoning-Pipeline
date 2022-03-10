import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from datasets import get_normalize_layer
from torch.nn import Softmax
import numpy as np

ARCHITECTURES = ['MLP','CNN']

def get_architecture(arch: str, dataset: str, classes: int = 10) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if  arch == "MLP":
        model = torch.nn.DataParallel(MLP(n_class = classes)).cuda()
    elif  arch == "CNN":
        model = torch.nn.DataParallel(CNN(n_class = classes)).cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
    
class MLP(nn.Module):
    def __init__(self, n_class = 10):
        super(MLP,self).__init__()
        
        if n_class == 10:
            self.dim = 28 * 28 * 5
        else:
            self.dim = 28 * 28

        hidden_1 = 512
        hidden_2 = 512
        self.droput = nn.Dropout(0.2)
            
        # linear layer (dim -> hidden_1)
        self.fc1 = nn.Linear(self.dim, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> n_class)
        self.fc3 = nn.Linear(hidden_2, n_class)

        
    def forward(self,x):
        # flatten image input
        x = x.view(-1, self.dim)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.droput(x)
         # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.droput(x)
        # add output layer
        x = self.fc3(x)
        return x
    
    
class CNN(nn.Module):
    def __init__(self, n_class = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, n_class)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output    # return x for visualization