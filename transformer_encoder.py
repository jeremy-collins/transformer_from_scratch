import torch
from torch import nn
from residual import Residual
from multi_headed_attention import MultiHeadedAttention
from feed_forward import feed_forward
from position_encoding import position_encoding

class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim=512, n_heads=6, ff_dim=2048, dropout=0.1):
        super().__init__()
        dim_q = max(model_dim // n_heads, 1)
        dim_k = max(model_dim // n_heads, 1)
        
        # first sublayer
        self.attention = Residual(MultiHeadedAttention(n_heads, model_dim, dim_q, dim_k), model_dim, dropout)
        # second sublayer
        self.feed_forward = Residual(feed_forward(model_dim, ff_dim, model_dim), model_dim, dropout)
    
    def forward(self, x):
        x = self.feed_forward(self.attention(x, x, x))
        return x
        
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, model_dim=512, n_heads=6, ff_dim=2048, dropout=0.1):
        super().__init__()
        
        # stacking num_layers of TransformerEncoderLayer
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(model_dim, n_heads, ff_dim, dropout) for layer in range(num_layers)]
        )
    
    def forward(self, x):
        seq_len = x.size(1)
        dims = x.size(2)
        
        x += position_encoding(seq_len, dims)
        
        # applying the encoder layers sequentially
        for layer in self.layers:
            x = layer(x)
        
        return x