import torch
import torch.nn as nn

import torchvision
import torchvision.datasets
import torchvision.transforms

import torch.optim
from torch.utils.data import dataloader

import time

class FLOWERS(nn.Module):
    def __init__(self):
        super(FLOWERS, self).__init__()
        self.name = "FLOWERS"
        self.vgg19 = torchvision.models.vgg19(pretrained = True)

        for parameter in self.vgg19.parameters():
            # print(parameter)
            parameter.requires_grad = False

        # print(self.vgg19.classifier)

        self.vgg19.classifier = nn.Sequential(
            nn.Linear(25088, 128),
            nn.ReLU(),

            nn.Linear(128, 5)
        )
    def forward(self, x):
        x = self.vgg19(x)
        return x
    
# if __name__== '__main__':
#     model = FLOWERS()
