import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super(RoPE, self).__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb.to(x.device)

        cos_emb, sin_emb = emb.cos(), emb.sin()
        cos_emb, sin_emb = cos_emb.unsqueeze(1), sin_emb.unsqueeze(1)

        x_rotated = torch.empty_like(x)
        for i in range(x.shape[1]):
            x_rotated[:, i] = self.rotate_half(x[:, i]) * cos_emb[i] + self.rotate_half(x[:, i], reverse=True) * sin_emb[i]

        return x_rotated

    def rotate_half(self, x, reverse=False):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        if reverse:
            return torch.cat((-x2, x1), dim=-1)
        return torch.cat((x2, -x1), dim=-1)
    
    
class NTKScalingRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, scaling_factor=1.0):
        super(NTKScalingRotaryEmbedding, self).__init__()
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb.to(x.device)

        # NTK Scaling 적용
        emb = emb * self.scaling_factor

        cos_emb, sin_emb = emb.cos(), emb.sin()
        cos_emb, sin_emb = cos_emb.unsqueeze(1), sin_emb.unsqueeze(1)

        x_rotated = torch.empty_like(x)
        for i in range(x.shape[1]):
            x_rotated[:, i] = self.rotate_half(x[:, i]) * cos_emb[i] + self.rotate_half(x[:, i], reverse=True) * sin_emb[i]

        return x_rotated

    def rotate_half(self, x, reverse=False):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        if reverse:
            return torch.cat((-x2, x1), dim=-1)
        return torch.cat((x2, -x1), dim=-1)	