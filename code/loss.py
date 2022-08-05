import torch
from torch.nn import Module


class CustomLoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, predict_y, y,p):

        temp = y - torch.mul(predict_y, p)

        return torch.mean(torch.square(temp))