import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn import Module


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.input_fc = nn.Linear(input_dim, 10)
        self.hidden_fc = nn.Linear(10, 10)
        self.output_fc = nn.Linear(10, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))
        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [batch size, 100]

        h3 = self.output_fc(h_2)
        # y_pred = [batch size, output dim]


        y_pred = F.sigmoid(h3)
        #y_pred = F.softmax(h3, dim=1)
        return y_pred, h3
    