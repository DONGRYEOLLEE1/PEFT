import torch
import pytest

from src.model import Transformer



bsz = 16
seq_len = 20
src_vocab_size = 1000
tgt_vocab_size = 1000

# 1. 모델 준비
@pytest.fixture(scope = "module")
def load_model():
    model = Transformer(src_vocab_size = src_vocab_size, tgt_vocab_size = tgt_vocab_size)
    return model

# 2. 간단한 연산
def test_inference(load_model):
    
    src = torch.randint(0, src_vocab_size, (bsz, seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (bsz, seq_len))
    
    out = load_model(src, tgt)
    
    print(out.shape)
    
    assert out.shape == torch.Size([bsz, seq_len, tgt_vocab_size])