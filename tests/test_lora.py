import torch
import pytest

from src.model import Transformer


@pytest.fixture
def model_factory():
    def factory(src_vocab_size, tgt_vocab_size):
        model = Transformer(src_vocab_size, tgt_vocab_size)
        return model
    return factory


@pytest.mark.parametrize(
    "src_vocab_size, tgt_vocab_size",
    [
        (1000, 1000),
    ]
)
def test_print_trainable_params(model_factory, src_vocab_size, tgt_vocab_size):
    
    model = model_factory(src_vocab_size, tgt_vocab_size)
    
    # 학습 가능한 파라미터 출력
    trainable_params = 0
    non_trainable_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            non_trainable_params += param.numel()

    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    
    assert trainable_params != 0
    
    
def test_lora_trainable_params():
    ...