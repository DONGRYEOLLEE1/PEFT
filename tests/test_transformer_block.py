import torch
import pytest

from src.model import Transformer


# 1. 모델 fixture: 파라미터를 받아 동적으로 생성
@pytest.fixture
def model(request):
    params = request.param
    return Transformer(
        src_vocab_size = params['src_vocab_size'],
        tgt_vocab_size = params['tgt_vocab_size']
    )

# 2. 테스트 함수에서 indirect 사용
@pytest.mark.parametrize(
    "batch_size, seq_len, model",
    [
        (16, 20, {"src_vocab_size": 1000, "tgt_vocab_size": 1000}),
        (32, 40, {"src_vocab_size": 2000, "tgt_vocab_size": 2000})
    ],
    indirect = ["model"],
)
def test_inference_on_transformer_block(
    batch_size, seq_len, model
):
    
    src_vocab_size = model.src_vocab_size
    tgt_vocab_size = model.tgt_vocab_size
    
    src = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))
    
    out = model(src, tgt)
    expected_out_shape = torch.Size([batch_size, seq_len, tgt_vocab_size])
    
    assert (
        out.shape == expected_out_shape,
        f"Expected shape {expected_out_shape}, but got {out.shape}"
    )