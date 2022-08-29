import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


"""
'D3' Architecture implementation, by Benker & al
https://www.sciencedirect.com/science/article/pii/S0278612520301928?via%3Dihub

+ Another architecture we tried
"""

class Conv(nn.Module):
    def __init__(self, win_length, n_features, activation='relu',
                dropout=0, bias=True, out_size=2):
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

        self.out_size = out_size

        if dropout > 0: 
           self.layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 9), bias=bias),
                nn.Dropout(p=dropout),
                act(),
                nn.Conv2d(16, 32, kernel_size=(2, 10), bias=bias),
                nn.Dropout(p=dropout),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Conv2d(32, 64, kernel_size=(2, 1), bias=bias),
                nn.Dropout(p=dropout),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Flatten(),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 9), bias=bias),
                act(),
                nn.Conv2d(16, 32, kernel_size=(2, 10), bias=bias),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Conv2d(32, 64, kernel_size=(2, 1), bias=bias),
                act(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Flatten(),
            )
        self.last = nn.Linear(
            64 * int((int((win_length - 5) / 2) - 1) / 2) * (n_features - 17), 
            out_size
        )
        self.softmax = nn.Softmax(dim=1)
        
        self.thresh = nn.Threshold(1e-9, 1e-9)
            
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cuda:0')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    def forward(self, x):
        if self.out_size <= 2: 
            x = F.softplus(self.last(self.layers(x.unsqueeze(1))))
            x = self.thresh(x)
            return x
        elif self.out_size > 2: 
            return self.softmax(self.last(self.layers(x.unsqueeze(1))))
        
class Conv2(nn.Module):
    def __init__(self, win_length, n_features, activation='relu',
                dropout=0, bias=True, out_size=2):
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

        self.out_size = out_size
        
        self.softmax = nn.LogSoftmax(-1)
        if dropout > 0:
            self.layers = nn.Sequential(
                nn.Conv2d(1, 32, 4),
                nn.Dropout(p = dropout),
                act(),
                nn.Conv2d(32, 32, 4),
                nn.Dropout(p=dropout),
                act(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(
                    int((win_length - 6) / 2) * int((n_features - 6) / 2) * 32, 
                    128
                ),
                nn.Dropout(p = dropout),
                act()
            )
        else: 
            self.layers = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding='same'),
                act(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 5, padding='same'),
                act(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(
                    int(int((win_length + 4 - 4) / 2 + 4 - 4) / 2) 
                    * int(int((n_features + 4 - 4) / 2 + 4 - 4) / 2) * 32, 
                    1024
                ),
                act()
            )
        self.last = nn.Linear(1024, self.out_size)
        self.thresh = nn.Threshold(1e-9, 1e-9)

    def save(self, path: str) -> None:
        """ Saves its state dict at path """
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cuda:0')):
        """ Loads the state dict at path """
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    def forward(self, x):
        if self.out_size <= 2: 
            x = F.softplus(self.last(self.layers(x.unsqueeze(1))))
            x = self.thresh(x)
            return x
        elif self.out_size > 2: 
            return self.softmax(self.last(self.layers(x.unsqueeze(1))))
        

if __name__ == "__main__":
    dnn = Conv(50, 18)
    print(summary(dnn, (100, 50, 18)))