import torch
from torch import nn
from transformer import Transformer

src = torch.randn(64, 32, 512)
tgt = torch.randn(64, 16, 512)

output = Transformer()(src, tgt)

print(output.shape)