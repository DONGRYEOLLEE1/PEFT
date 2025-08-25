import torch

from src.model import MHA


if __name__ == "__main__":
    
    batch_size = 2
    seq_len = 8
    
    
    dummy_x = torch.randn(batch_size, seq_len, 4)
    
    attn = MHA(
        d_model = 4,
        num_heads = 2,
        dropout = .1
    )
    
    out_attn = attn(dummy_x, dummy_x, dummy_x)
    
    print(f"NORMAL ATTENTION OUT SHAPE: {out_attn.shape}")
    
    
    attn = MHA(
        d_model = 4,
        num_heads = 2,
        dropout = .1,
        rope_type = "rope",
        rope_dim = 2,
        rope_base = 10000
    )
    
    out_rope = attn(dummy_x, dummy_x, dummy_x)
    
    print(f"OUT ROPE SHAPE: {out_rope.shape}")