import torch
import torch.nn as nn
import torch.nn.functional as F

# Model architectures based on:
# https://github.com/kkangshen/bayesian-deep-rul/blob/master/models/
# (Just model examples to be assessed and modified according to our needs)
class Linear(nn.Module):
    def __init__(self, win_length, n_features, activation='relu', 
                dropout_freq=0, bias=True, typ="regression"):
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

        self.typ = typ
        if typ == "regression": out_size = 2
        elif typ == "classification": out_size = 10
        else: raise ValueError(f"Unknown value for typ : {typ}")

        if dropout_freq > 0 :
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(win_length * n_features, 256, bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(256, 128, bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(128, 128, bias=bias),
                nn.Dropout(p=dropout_freq),
                act(),
                nn.Linear(128, 32, bias=bias),
                nn.Dropout(p=dropout_freq),
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

    def save(self, path: str) -> None:
        torch.save(self.state_dict(),path)

    def load(self, path: str, map_location = torch.device('cuda:0')):
        state_dict = torch.load(path,map_location=map_location)
        self.load_state_dict(state_dict)

    def forward(self, x):
        if self.typ == "regression": 
            return 1e-9 + 100 * F.softplus(self.last(self.layers(x.unsqueeze(1))))
        elif self.typ == "classification": 
            return self.softmax(self.last(self.layers(x.unsqueeze(1))))
