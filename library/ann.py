import torch

from torch import nn

class ANN(nn.Module):

    def __init__(self, n_inputs) -> None:
        super().__init__()

        self.hidden1 = nn.Linear(n_inputs, 128)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(128, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        y = self.act1(self.hidden1(x))
        y = self.act2(self.hidden2(y))
        return self.act_output(self.output(y))
