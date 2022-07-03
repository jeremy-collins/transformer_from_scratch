from turtle import forward
import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, sublayer, dims, dropout=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dims)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, *inputs):
        # inputs = query, key, value
        # norm(query + dropout(sublayer(query, key, value)))
        return self.norm(inputs[0] + self.dropout(self.sublayer(*inputs)))