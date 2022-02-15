import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        # Data Loading
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples


batch_size = 4
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

epochs = 2
num_samples = len(dataset)
n_iters = math.ceil(num_samples/batch_size)
print(num_samples, n_iters)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            print(f'Epoch {(epoch+1)}/{epochs}, step {(i+1)}/{n_iters}, inputs {inputs.shape}')
