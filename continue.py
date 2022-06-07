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
model.state_dict(torch.load( open( "model.p", "rb" ) ))

#Initialize the optimizer
optimizerAdam = torch.optim.Adam(model.parameters())

out_dictAdam = train(dataset, model, optimizerAdam, 10)

pickle.dump(model, open( "model2.p", "wb" ) )