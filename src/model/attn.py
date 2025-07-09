import math
import torch
import torch.nn as nn

from typing import Optional, Literal

from .positional_encodings import RotaryEmbedding, NTKScalingRotaryEmbedding, YaRNRotaryEmbedding


class MHA(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_type: Optional[Literal["rope", "ntk_rope", "yarn"]] = None,
        rope_dim: Optional[int] = None, 
        rope_base: Optional[int] = 10000,
        scaling_factor: Optional[float] = 1.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f'{d_model} % {num_heads} != 0'
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        self.k_proj = nn.Linear(d_model, d_model, bias = False)
        self.v_proj = nn.Linear(d_model, d_model, bias = False)
        self.o_proj = nn.Linear(d_model, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE 초기화
        if rope_type == "rope":
            self.rope = RoPE(dim = rope_dim, base = rope_base)
        elif rope_type == "ntk_rope":
            self.rope = NTKScalingRotaryEmbedding(dim = rope_dim, base = rope_base, scaling_factor = scaling_factor)
        elif rope_type == "yarn":
            ...
        else:
            self.rope = None
        
    def _transform(self, batch_size, x, linear):
        x = linear(x)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        
        return x.transpose(1, 2).contiguous()
        
    def forward(self, q, k, v, mask=None):
        # batch_size = q.size(0)
        batch_size, seq_len, _ = q.size()
        
        # 1. Linear projections & split by heads
        q = self._transform(batch_size, q, self.q_proj)
        k = self._transform(batch_size, k, self.k_proj)
        v = self._transform(batch_size, v, self.v_proj)
        
        # 2. Apply RoPE if provided
        if self.rope is not None:
            # RoPE는 각 헤드에 대해 적용되어야 함
            # seq_len = q.size(2)
            
            q_reshaped = q.reshape(batch_size * self.num_heads, seq_len, self.d_k)
            k_reshaped = k.reshape(batch_size * self.num_heads, seq_len, self.d_k)
            
            q = self.rope(q_reshaped, seq_len).view(batch_size, self.num_heads, seq_len, self.d_k)
            k = self.rope(k_reshaped, seq_len).view(batch_size, self.num_heads, seq_len, self.d_k)
        
        # 3. Scaled dot-product attn
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e3)
            
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)     # bacth, num_heads, seq_len, d_k
        
        # 4. Concat heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.o_proj(out)
        
        return out