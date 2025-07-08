import torch
import torch.nn as nn

from .attn import MHA
from .ffn import FFN
from .positional_encodings import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = .1):
        super().__init__()
        self.attn = MHA(d_model, num_heads, dropout)
        self.cross_attn = MHA(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, memory, src_mask = None, tgt_mask = None):
        x2 = self.norm1(x + self.dropout1(self.attn(x, x, x, tgt_mask)))
        x3 = self.norm2(x2 + self.dropout2(self.cross_attn(x2, memory, memory, src_mask)))
        x4 = self.norm3(x3 + self.dropout3(self.ffn(x3)))
        
        return x4
    
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout = .1, max_len = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tgt, memory, src_mask = None, tgt_mask = None):
        x = self.embedding(tgt)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        
        return x