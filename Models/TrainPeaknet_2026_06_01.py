import torch.nn as nn


class Peaknet(nn.Module):
    def __init__(self):
        # This is the same model architecture used in TrainPeaknet.ipynb.
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=51, padding=25),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=51, padding=25),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=51, padding=25),
            # Sigmoid squashes the output into [0, 1] to match the mask range
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# Lowercase alias so older code that expects peaknet() can still use this model.
peaknet = Peaknet
