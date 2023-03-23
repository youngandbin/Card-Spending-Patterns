import torch.nn as nn
from torch.utils.data import Dataset
import os
import errno

class creditDataloader(Dataset):
    def __init__(self, csv):

        self.data = csv.loc[:, '가정용품':'주거']
        self.data = self.data.to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Autoencoder(nn.Module):
    def __init__(self, numLayers, encoders=False):

        super().__init__()
        self.layers = nn.ModuleList()

        if encoders:
            for i in range(len(numLayers) - 2):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[-2], numLayers[-1]))
        else:
            for i in range(len(numLayers) - 2):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[-2], numLayers[-1]))
            for i in range(len(numLayers) - 1, 1, -1):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i-1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[1], numLayers[0]))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        return y

