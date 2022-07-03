import torch
from torch import nn

def position_encoding(seq_len, dim_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float32, device=device).reshape(1, 1, -1)
    
    phase = pos / (1e4 ** (torch.div(dim, dim_model, rounding_mode='trunc')))
    
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))