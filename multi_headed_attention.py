import torch
from torch import nn
from attention_head import AttentionHead

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_q, dim_k):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for h in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)
        
    def forward(self, query, key, value):
        # feeding the concatenation of all attention heads through a linear layer
        return self.linear(torch.cat([head(query, key, value) for head in self.attention_heads], dim=-1))