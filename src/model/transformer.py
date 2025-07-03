import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        tgt_vocab_size, 
        d_model = 512, 
        num_layers = 6, 
        num_heads = 8, 
        d_ff = 2048,
        dropout = .1,
        max_len = 5000
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            dropout,
            max_len
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            dropout,
            max_len
        )
        self.out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        memory = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, memory, src_mask, tgt_mask)
        out = self.out(dec_out)
        
        return out