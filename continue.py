import pickle

import torch
from second import Network
from train import train

import hotdog

from second import Network


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = hotdog.dataset()

model = Network() 
model.load_state_dict(torch.load("model.p"))

#Initialize the optimizer
optimizerAdam = torch.optim.Adam(model.parameters())

out_dictAdam = train(dataset, model, optimizerAdam, 10)

torch.save( model.state_dict(), "model2.p")