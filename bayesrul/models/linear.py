import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

# Model architectures based on:
# https://github.com/kkangshen/bayesian-deep-rul/blob/master/models/
# (Just model examples to be assessed and modified according to our needs)
class Linear(nn.Module):
    def __init__(self, win_length, n_features, activation='relu', 
                dropout=0, bias=True, out_size=1):
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

        if dropout > 0 :
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(win_length * n_features, 256, bias=bias),
                nn.Dropout(p=dropout),
                act(),
                nn.Linear(256, 128, bias=bias),
                nn.Dropout(p=dropout),
                act(),
                nn.Linear(128, 128, bias=bias),
                nn.Dropout(p=dropout),
                act(),
                nn.Linear(128, 32, bias=bias),
                nn.Dropout(p=dropout),
                act(),
            )
        else:
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(win_length * n_features, 256, bias=bias),
                act(),
                nn.Linear(256, 128, bias=bias),
                act(),
                nn.Linear(128, 128, bias=bias),
                act(),
                nn.Linear(128, 32, bias=bias),
                act(),
            )
        self.last = nn.Linear(32, out_size)
        self.softmax = nn.Softmax(dim=1)
        self.thresh = nn.Threshold(1e-9, 1e-9)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(),path)

    def load(self, path: str, map_location = torch.device('cuda:0')):
        state_dict = torch.load(path,map_location=map_location)
        self.load_state_dict(state_dict)

    def forward(self, x):
        if self.out_size <= 2: 
            x = F.softplus(self.last(self.layers(x.unsqueeze(1))))
            x = self.thresh(x)
            return x
        elif self.out_size > 2: 
            return self.softmax(self.last(self.layers(x.unsqueeze(1))))
        


if __name__ == "__main__":
    dnn = Linear(50, 18)
    print(summary(dnn, (100, 50, 18)))