import torch
from torch import nn
from residual import Residual
from multi_headed_attention import MultiHeadedAttention
from feed_forward import feed_forward
from position_encoding import position_encoding

class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim=512, n_heads=6, ff_dim=2048, dropout=0.1):
        super().__init__()
        dim_q = max(model_dim // n_heads, 1)
        dim_k = max(model_dim // n_heads, 1)
        
        # first sublayer
        self.attention_1 = Residual(MultiHeadedAttention(n_heads, model_dim, dim_q, dim_k), model_dim, dropout)
        # second sublayer
        self.attention_2 = Residual(MultiHeadedAttention(n_heads, model_dim, dim_q, dim_k), model_dim, dropout)
        # third sublayer
        self.feed_forward = Residual(feed_forward(model_dim, ff_dim, model_dim), model_dim, dropout)
        
    def forward(self, target, memory):
        target = self.attention_1(target, target, target)
        target = self.attention_2(target, memory, memory)
        return self.feed_forward(target)
    
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, model_dim=512, n_heads=6, ff_dim=2048, dropout=0.1):
        super().__init__()
        # stacking num_layers of TransformerDecoderLayer
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(model_dim, n_heads, ff_dim, dropout) for layer in range(num_layers)]
        )
        self.linear = nn.Linear(model_dim, model_dim)
        
    def forward(self, target, memory):
        seq_len = target.size(1)
        dims = target.size(2)
        
        target += position_encoding(seq_len, dims)
        
        # applying the decoder layers sequentially
        for layer in self.layers:
            target = layer(target, memory)
        
        # pushing the decoder output through a linear layer and normalizing
        # the outputs are probabilities of tokens in the vocabulary
        return torch.softmax(self.linear(target), dim=1)       