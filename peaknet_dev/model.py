import torch
import torch.nn as nn


class Peaknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=27, padding=13),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=27, padding=13),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=27, padding=13),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


peaknet = Peaknet
