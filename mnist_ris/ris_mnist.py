# -*- coding: utf-8 -*-
"""lightining.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RLK925jaoq1pig4yUEMzy4st-WLn1GEO
"""

import torch
from torch import nn
from torch import optim
import numpy as np

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pdb
import pytorch_lightning as pl

from torchmetrics.functional import accuracy



# setting device
device = torch.device('cuda' if torch.cuda.is_available() is True else 'cpu')
print(f'Current device is {device}')

"""## Pytorch Lightning

1) Model

2) Optimizer

3) Data

4) Training Loop "the magic"

5) Validation Loop
"""

# sanity check for gpu
print('Sanity check for cuda')
torch.rand(1, device='cuda:0')
print('Passed')
"""## Updates with the pl.LightningModule 

#### 1) configure_optimizers(self):
        Define the optimizer in this method

#### 2) training_step(self,batch,batch_idx)
        a) No need to use [.to(device)]. Pytorch figures it out by itself

        b) Only need to define the forward step and compute objective. No need to set gradient to zero, calculate gradients and update weights manually

#### 3) train_dataloader(self)
        You can load the data inside the class. This gives the model access to the data and any attributes of the data it might need. For eg: lenght of data

"""

#functional model

class ResNet(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(28*28,64)
    self.l2 = nn.Linear(64,64)
    self.l3 = nn.Linear(64,10)
    self.drop = nn.Dropout(0.1)

    self.ce_loss = nn.CrossEntropyLoss()

  def forward(self,x):
    h1 = nn.functional.relu(self.l1(x))
    h2 = nn.functional.relu(self.l2(h1))
    do = self.drop(h2 + h1)
    logit = self.l3(do)
    return logit

  def configure_optimizers(self):
    optimizer = optim.SGD(self.parameters(), lr=1e-2)
    return optimizer

  def prepare_data(self):
    #download the data
    torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

  def setup(self, stage=None):
    #split and transforms here
    self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    self.train_data, self.val_data = random_split(self.train_dataset, [55000,5000])

  def shared_eval_step(self, batch, batch_idx):
    x,y = batch

    batch_size = x.size(0)
    x = x.view(batch_size,-1)

    # forward
    logits = self(x)

    # objective
    loss = self.ce_loss(logits, y)

    acc = accuracy(logits, y)

    return(loss,acc)


  def training_step(self, batch, batch_idx):
    J, acc = self.shared_eval_step(batch, batch_idx)
    self.log_dict({'train_loss':J,'acc':acc}, prog_bar=True, logger=True, on_epoch=True, on_step=False)
    return {'loss':J}

  def validation_step(self, batch,batch_idx):
    J, acc = self.shared_eval_step(batch, batch_idx)
    self.log_dict({'val_loss':J, 'val_acc':acc}, prog_bar=True, logger=True, on_epoch=True, on_step=False)
    return {'loss':J}

  def test_step(self, batch, batch_idx):
    J, acc = self.shared_eval_step(batch, batch_idx)
    self.log_dict({'test_loss':J, 'test_acc':acc}, prog_bar=True, logger=True, on_epoch=True, on_step=False)
    return {'loss':J}

  def train_dataloader(self):
    batch_size = 32
    train_loader  = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True, num_workers=20)
    return train_loader

  def val_dataloader(self):
    batch_size = 32
    val_loader  = DataLoader(dataset=self.val_data, batch_size=batch_size, shuffle=False, num_workers=20)
    return val_loader

  def test_dataloader(self):
    batch_size = 32
    test_loader  = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    return test_loader


if __name__ == '__main__':
    model = ResNet()

    """### The Trainer
    The training is controlled by the Trainer. You specify all the training
    parameters like max_epochs, acceleration, DataLoaders, number of gpu etc in the
    trainer and use the trainer to fit your model.
    """

    trainer = pl.Trainer(max_epochs=4, accelerator='gpu')
    trainer.fit(model)

    print(trainer.test())

