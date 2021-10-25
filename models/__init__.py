from typing import ForwardRef
import torch
import torch.nn as nn


class CrossModalMatchingHead(nn.Module):
    def __init__(self, num_classes, feats_dim):
        super(CrossModalMatchingHead, self).__init__()
        self.label_embedding = nn.Linear(num_classes, 128)
        self.mlp = nn.Sequential(
            nn.Linear(feats_dim + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, y):
        # y is onehot vectors
        y_embbeding = self.label_embedding(y)
        return self.mlp(torch.cat([x, y_embbeding], dim=1))
