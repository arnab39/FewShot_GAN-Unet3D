"""
An example for loss class definition, that will be used in the agent
"""
import torch.nn as nn


class CrossEntropyLoss3d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss

# Dice loss
