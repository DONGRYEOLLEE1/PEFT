import torch
from src.model.positional_encodings import YaRNRotaryEmbedding

batch_size = 2
seq_len = 2048
dim = 128

yarn = YaRNRotaryEmbedding(
    dim = dim,
    base = 10000,
    scaling_factor = 1.0,
    original_max_position_embeddings= 1024,
    extrapolation_factor = 1.0,
    attn_factor = 1.0,
    beta_fast = 32,
    beta_slow = 1
)

x = torch.randn(batch_size, seq_len, dim)

# YaRN RoPE 적용
x_rotated = yarn(x)

print(f"입력 크기: {x.shape}")
print(f"출력 크기: {x_rotated.shape}")
print(f"원래 최대 길이: {yarn.original_max_position_embeddings}")
print(f"현재 시퀀스 길이: {seq_len}")
print(f"스케일링 비율: {seq_len / yarn.original_max_position_embeddings:.2f}")