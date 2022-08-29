# Architecture from: Nathaniel DeVol, Christopher Saldana, Katherine Fu
# https://papers.phmsociety.org/index.php/phmconf/article/view/3109

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class InceptionModule(nn.Module):
    def __init__(
        self, 
        n_features,
        filter1x1, 
        filter3x3, 
        filter5x5, 
        filterpool,
        activation: nn.Module,
        bias = True,
        dropout = 0,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, filter1x1, kernel_size=1, padding='same', bias=bias),
            activation()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(n_features, filter3x3, kernel_size=3, padding='same', bias=bias),
            activation()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(n_features, filter5x5, kernel_size=5, padding='same', bias=bias),
            activation()
        )
        self.convpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(n_features, filterpool, kernel_size=3, padding='same', bias=bias),
            activation()
        )
        if dropout > 0:
            self.conv1.add_module('branch1_dropout', nn.Dropout(dropout/4))
            self.conv3.add_module('branch2_dropout', nn.Dropout(dropout/4))
            self.conv5.add_module('branch3_dropout', nn.Dropout(dropout/4))
            self.convpool.add_module('branch4_dropout', nn.Dropout(dropout/4))
            
    
    def forward(self, x):
        branches = [self.conv1(x), self.conv3(x), self.conv5(x), self.convpool(x)]
        return torch.cat(branches, 1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cpu')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)


class InceptionModuleReducDim(nn.Module):
    def __init__(
        self, 
        n_features,
        filter1x1, 
        reduc3x3,
        filter3x3, 
        reduc5x5,
        filter5x5, 
        filterpool,
        activation: nn.Module,
        bias = True,
        dropout = 0,
    ):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(n_features, filter1x1, kernel_size=1, padding='same', bias=bias),
            activation()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(n_features, reduc3x3, kernel_size=1, padding=0, bias=bias),
            activation(),
            nn.Conv1d(reduc3x3, filter3x3, kernel_size=3, padding='same', bias=bias),
            activation()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(n_features, reduc5x5, kernel_size=1, padding=0, bias=bias),
            activation(),
            nn.Conv1d(reduc5x5, filter5x5, kernel_size=5, padding='same', bias=bias), 
            activation()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(n_features, filterpool, kernel_size=1, padding=0, bias=bias),
            activation()
        )
        if dropout > 0:
            self.branch1.add_module('branch1_dropout', nn.Dropout(dropout/4))
            self.branch2.add_module('branch2_dropout', nn.Dropout(dropout/4))
            self.branch3.add_module('branch3_dropout', nn.Dropout(dropout/4))
            self.branch4.add_module('branch4_dropout', nn.Dropout(dropout/4))
    
    def forward(self, x):
        branches = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(branches, 1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cpu')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

class InceptionModel(nn.Module):
    """
    Adapted version of Inception model below to have less parameters
    """
    def __init__(self, win_length, n_features, 
            dropout = 0, activation='relu', bias='True', out_size=2):
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

        self.layers = nn.Sequential(
            InceptionModule(n_features,
                27, 27, 27, 27, activation = act, bias = bias, dropout=dropout
            ),
            InceptionModuleReducDim(
                27+27+27+27, 8, 64, 8, 64, 8, 32, activation = act, bias = bias,
                dropout=dropout,
            ),
            InceptionModuleReducDim(
                8+8+8+32, 4, 16, 4, 16, 4, 8, activation = act, bias = bias,
                dropout=dropout,
            ),
            nn.Flatten(),
            nn.Linear((4 + 4 + 4 + 8) * win_length, 64),
            act(), 
        )
        if dropout > 0:
            self.layers.add_module('n-1_dropout', nn.Dropout(dropout))

        self.last = nn.Linear(64, self.out_size)
        self.thresh = nn.Threshold(1e-9, 1e-9)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        if self.out_size <= 2: 
            x = F.softplus(self.last(self.layers(x.transpose(2, 1))))
            x = self.thresh(x)
            return x
        elif self.out_size > 2:
            return self.softmax(self.last(self.layers(x.transpose(2, 1))))
        

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cpu')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)


class BigCeption(nn.Module):
    """
    Inception model of DeVol & Al
    https://papers.phmsociety.org/index.php/phmconf/article/view/3109
    """
    def __init__(self, win_length, n_features, activation='relu', bias='True',
            dropout=0, out_size=2):
        super().__init__()
        assert n_features == 18, \
            "TODO, Generalize Inception model for other than 18 features"

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

        self.layers = nn.Sequential(
            InceptionModule(n_features, 27, 27, 27, 27, activation = act, 
                    dropout=dropout, bias = bias),
            InceptionModuleReducDim(27+27+27+27, 16, 64, 16, 64, 16, 32, 
                    dropout=dropout, activation = act, bias = bias),
            nn.Flatten(),
            nn.Linear((16+16+16+32) * win_length, 64),
            act(), 
        )
        if dropout > 0:
            self.layers.add_module('n-1_dropout', nn.Dropout(dropout))


        self.last = nn.Linear(64, self.out_size)
        self.thresh = nn.Threshold(1e-9, 1e-9)
        
    def forward(self, x):
        if self.out_size <= 2: 
            x = F.softplus(self.last(self.layers(x.transpose(2, 1))))
            x = self.thresh(x)
            return x
        elif self.out_size > 2:
            return self.softmax(self.last(self.layers(x.transpose(2, 1))))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cpu')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

if __name__ == "__main__":
    dnn = BigCeption(30, 18)
    print(summary(dnn, (100, 30, 18)))
    dnn = InceptionModel(30, 18)
    print(summary(dnn, (100, 30, 18)))