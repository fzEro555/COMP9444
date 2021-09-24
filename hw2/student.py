#!/usr/bin/env python3
"""
Jia (Andrew) Wu [z5163239], Yifan He [z5173587]
5/8/21

student.py

UNSW COMP9444 Neural Networks and Deep Learning
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from config import device

"""
    SEE PDF FOR ANSWERS TO QUESTIONS.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Image transformations for Data Augmentation during training.

    Only Grayscale and ToTensor are applied during testing to
    ensure model classifies the original images.
    """
    if mode == 'train':
        return transforms.Compose(
            [
            transforms.Grayscale(), 
            transforms.RandomResizedCrop((64, 64), scale=(0.5, 1.0)),
            transforms.RandomPerspective(p=0.2),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.0, 0.5)),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor()
            ]
        )
    elif mode == 'test':
        return transforms.Compose(
            [
            transforms.Grayscale(),
            transforms.ToTensor(),
            ]
        )


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    """
    Network class for final model. 

    Architecture:
        1. Convolutional and maxpooling blocks
            7 Conv2D layers, 4 MaxPool2D layers with
            BatchNorm, ReLU activation
        2. Adapted Average Pooling Layer
        3. Fully connected layers
            3 dense layers with BatchNorm, ReLU activation,
            and dropout.
    """
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5,5))
        self.linear_layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(192*5*5, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 14),
            nn.BatchNorm1d(14)
        )

    def forward(self, t):
        x = self.cnn_layers(t)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_layers(x)
        return x


net = Network()
net.to(device)
lossFunc = nn.CrossEntropyLoss()


############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 1
batch_size = 64
epochs = 100
optimiser = optim.Adam(net.parameters(), lr=0.0006)
