import torch
from torch import nn
from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers=6,
        num_decoder_layers=6,
        model_dim=512,
        n_heads=6,
        ff_dim=2048,
        dropout=0.1,
        activation = nn.ReLU
        ):
        super().__init__()
        
        self.encoder = TransformerEncoder(num_layers=num_encoder_layers,
                                          model_dim=model_dim,
                                          n_heads=n_heads,
                                          ff_dim=ff_dim,
                                          dropout=dropout
                                          )
        
        self.decoder = TransformerDecoder(num_layers=num_decoder_layers,
                                          model_dim=model_dim,
                                          n_heads=n_heads,
                                          ff_dim=ff_dim,
                                          dropout=dropout)
        
    def forward(self, encoder_input, decoder_input):
        # 4+ levels of abstraction later...
        return self.decoder(decoder_input, self.encoder(encoder_input))