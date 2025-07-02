import torch
import torch.nn as nn

import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = .1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))