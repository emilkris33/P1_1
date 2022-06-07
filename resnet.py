
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

drop_rate = 0.5

class ResNetBlock(nn.Module):
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()
        self.convolutional = nn.Sequential(
                nn.Conv2d(n_features, n_features, 3, stride=1, padding='same'),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(),
                nn.Conv2d(n_features, n_features, 3, stride=1, padding='same'),
                nn.Dropout2d(p=drop_rate)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.convolutional(x) + x
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, n_in, n_features, num_res_blocks=3):
        super(ResNet, self).__init__()
        #First conv layers needs to output the desired number of features.
        conv_layers = [ nn.Conv2d(n_in, n_features, kernel_size=7, stride=1, padding=1),
                        nn.Dropout2d(p=drop_rate),
                        nn.ReLU()]
        for i in range(num_res_blocks):
            conv_layers.append(ResNetBlock(n_features))
        self.res_blocks = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(nn.Linear(15376*n_features, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512,10),
                                nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        x = self.res_blocks(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

modelAdam = ResNet(3, 16, 10)
modelAdam.to(device)
#Initialize the optimizer
optimizerAdam = torch.optim.Adam(modelAdam.parameters())

out_dictAdam = train(dataset, modelAdam, optimizerAdam, 10)