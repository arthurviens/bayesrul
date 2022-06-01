# Architecture from DeVol, Saldana, Fu
# https://papers.phmsociety.org/index.php/phmconf/article/view/3109

import torch
import torch.nn as nn
from torchinfo import summary


class InceptionModule(nn.Module):
    def __init__(
        self, 
        in_channels, 
        filter1x1, 
        filter3x3, 
        filter5x5, 
        filterpool,
        activation: nn.Module,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, filter1x1, kernel_size=1, padding='same'),
            activation()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, filter3x3, kernel_size=3, padding='same'),
            activation()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, filter5x5, kernel_size=5, padding='same'),
            activation()
        )
        self.convpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, filterpool, kernel_size=3, padding='same'),
            activation()
        )
    
    def forward(self, x):
        branches = [self.conv1(x), self.conv3(x), self.conv5(x), self.convpool(x)]
        return torch.cat(branches, 1)


class InceptionModuleReducDim(nn.Module):
    def __init__(
        self, 
        in_channels, 
        filter1x1, 
        reduc3x3,
        filter3x3, 
        reduc5x5,
        filter5x5, 
        filterpool,
        activation: nn.Module,
    ):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, filter1x1, kernel_size=1, padding='same'),
            activation()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, reduc3x3, kernel_size=1, padding=0),
            activation(),
            nn.Conv1d(reduc3x3, filter3x3, kernel_size=3, padding='same'),
            activation()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, reduc5x5, kernel_size=1, padding=0),
            activation(),
            nn.Conv1d(reduc5x5, filter5x5, kernel_size=5, padding='same'), 
            activation()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, filterpool, kernel_size=1, padding=0),
            activation()
        )
    
    def forward(self, x):
        branches = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(branches, 1)



class InceptionModel(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()

        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'sigmoid':
            act = nn.Sigmoid
        elif activation == 'tanh':
            act = nn.Tanh
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU
        else:
            raise ValueError("Unknown activation")

        self.layers = nn.Sequential(
            InceptionModule(18, 27, 27, 27, 27, activation = act),
            InceptionModuleReducDim(108, 32, 64, 32, 64, 32, 32, activation = act),
            nn.Flatten(),
            nn.Linear(3840, 256),
            nn.Linear(256, 1)
        )
        
    
    def forward(self, x):
        return self.layers(x)

