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
    
    def forward(self, x):
        branches = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(branches, 1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cpu')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

class InceptionModel(nn.Module):
    def __init__(self, win_length, n_features, activation='relu', bias='True'):
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

        self.out_size = 2 

        self.layers = nn.Sequential(
            InceptionModule(n_features,
                27, 27, 27, 27, activation = act, bias = bias
            ),
            InceptionModuleReducDim(
                27+27+27+27, 8, 64, 8, 64, 8, 32, activation = act, bias = bias
            ),
            InceptionModuleReducDim(
                8+8+8+32, 4, 16, 4, 16, 4, 8, activation = act, bias = bias
            ),
            nn.Flatten(),
            nn.Linear((4 + 4 + 4 + 8) * win_length, 64),
            act(), 
        )
        self.last = nn.Linear(64, self.out_size)

        self.thresh = nn.Threshold(1e-9, 1e-9)
        
    
    def forward(self, x):
        x = F.softplus(self.last(self.layers(x.transpose(2, 1))))
        x = self.thresh(x)
        return x

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cpu')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)


class BigCeption(nn.Module):
    def __init__(self, win_length, n_features, activation='relu', bias='True'):
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

        self.out_size = 2 

        self.layers = nn.Sequential(
            InceptionModule(n_features, 27, 27, 27, 27, activation = act, bias = bias),
            InceptionModuleReducDim(27+27+27+27, 16, 64, 16, 64, 16, 32, activation = act, bias = bias),
            nn.Flatten(),
            nn.Linear((16+16+16+32) * win_length, 64),
            act(), 
        )
        self.last = nn.Linear(64, self.out_size)
        self.thresh = nn.Threshold(1e-9, 1e-9)
        
    
    def forward(self, x):
        x = F.softplus(self.last(self.layers(x.transpose(2, 1))))
        x = self.thresh(x)
        return x

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location = torch.device('cpu')):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

if __name__ == "__main__":
    dnn = BigCeption(50, 18)
    print(summary(dnn, (100, 50, 18)))