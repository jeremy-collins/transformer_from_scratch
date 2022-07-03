import torch
import torch.nn as nn

def feed_forward(input_dim=512, hidden_dim=2048, output_dim=512):
    # The feed forward sublayer that's present at the end of the encoder and the end of the decoder
    # Has a residual connection around it
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )