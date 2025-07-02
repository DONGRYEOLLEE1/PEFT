import setup_path

from src.model import Transformer

import torch


bsz = 16
seq_len = 20
src_vocab_size = 1000
tgt_vocab_size = 1000

model = Transformer(src_vocab_size = src_vocab_size, tgt_vocab_size = tgt_vocab_size)

src = torch.randint(0, src_vocab_size, (bsz, seq_len))
tgt = torch.randint(0, tgt_vocab_size, (bsz, seq_len))

out = model(src, tgt)

print(out.shape)

assert out.shape == torch.Size([bsz, seq_len, tgt_vocab_size])