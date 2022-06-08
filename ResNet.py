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
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from train import train
import hotdog


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = hotdog.dataset()


model_ADAM = models.resnet18(pretrained=True)
num_ftrs = model_ADAM.fc.in_features
model_ADAM.fc = nn.Linear(num_ftrs, 2)
model_ADAM.to(device)

ADAM_lr = 0.000005
optimizer_ADAM = torch.optim.Adam(model_ADAM.parameters(),lr=ADAM_lr)

out_dictAdam = train(dataset, model_ADAM, optimizer_ADAM, 10)