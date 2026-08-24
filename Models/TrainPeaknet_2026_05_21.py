import torch.nn as nn


class Peaknet(nn.Module):
    def __init__(self):
        # This is the same model architecture used in TrainPeaknet.ipynb.
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=51, padding=25),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=51, padding=25),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=51, padding=25),
        )

    def forward(self, x):
        return self.net(x)


# Lowercase alias so older code that expects peaknet() can still use this model.
peaknet = Peaknet
