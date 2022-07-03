import torch
from torch import nn
from scaled_dot_product_attention import scaled_dot_product_attention

class AttentionHead(nn.Module):
    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
        
    def forward(self, query, key, value):
        # the inputs are fed through the linear layers first
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))
        