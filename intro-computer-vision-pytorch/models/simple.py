import torch.nn as nn


class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__()
    self.seq = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28 * 28, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10),
      nn.LogSoftmax(dim=0),
    )

  def forward(self, x):
    return self.seq(x)
