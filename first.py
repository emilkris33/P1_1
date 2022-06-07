import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from train import train
import hotdog


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = hotdog.dataset()

drop_rate = 0.3

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
                nn.Conv2d(3, 8, 3, stride=1, padding='same'),
                nn.BatchNorm2d(8),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, stride=1, padding='same'),
                nn.BatchNorm2d(8),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, stride=1, padding='same'),
                nn.BatchNorm2d(16),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=1, padding='same'),
                nn.BatchNorm2d(16),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU())

        self.fully_connected = nn.Sequential(
                nn.Linear(65536, 500),
                nn.Dropout(p=drop_rate),
                nn.ReLU(),
                nn.Linear(500, 2),
                nn.LogSoftmax(dim=1))
    
    def forward(self, x):
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

modelAdam = Network()
modelAdam.to(device)
#Initialize the optimizer
optimizerAdam = torch.optim.Adam(modelAdam.parameters())

out_dictAdam = train(dataset, modelAdam, optimizerAdam, 10)