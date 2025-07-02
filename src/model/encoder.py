import torch
import torch.nn as nn

from .attn import MHA
from .ffn import FFN
from .pe import PositionalEncoding



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = .1):
        super().__init__()
        
        self.mha = MHA(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask = None):
        x2 = self.norm1(x + self.dropout1(self.mha(x, x, x, mask)))
        x3 = self.norm2(x2 + self.dropout2(self.ff(x2)))
        return x3
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout = .1, max_len= 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask = None):
        x = self.embedding(src)
        x = self.pe(x)
        
        if src_mask is not None:
            for layer in self.layers:
                x = layer(x, src_mask)
        x = self.norm(x)
        
        return x