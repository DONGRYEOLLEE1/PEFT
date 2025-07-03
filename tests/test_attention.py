import torch
import pytest

from src.model import MHA


@pytest.fixture
def mha_factory():
    def factory(d_model, num_heads, dropout):
        return MHA(d_model, num_heads, dropout)
    return factory


def test_mha_forward_shape(mha_factory):
    batch_size = 2
    seq_len = 8
    d_model = 32
    num_heads = 4
    dropout = .1
    
    x = torch.randn(batch_size, seq_len, d_model)
    mha = mha_factory(d_model = d_model, num_heads = num_heads, dropout = dropout)
    
    out = mha(x, x,  x)
    
    assert out.shape == (batch_size, seq_len, d_model)
    
    
def test_mha_invalid_heads(mha_factory):
    """MHA d_k param 체크"""
    with pytest.raises(AssertionError):
        mha_factory(d_model = 30, num_heads = 7, dropout = .1)
        
    mha_factory(d_model = 30, num_heads = 5, dropout = .1)


def test_mha_mask_effect(mha_factory):
    batch_size, seq_len, d_model, num_heads = 1, 4, 8, 2
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    mha = mha_factory(d_model, num_heads, .1)
    
    mask = torch.tensor([[1, 1, 0, 0]])
    mask = mask.unsqueeze(1).unsqueeze(2)   # (batch_size, 1, 1, seq_len)
    
    out_with_mask  = mha(x, x, x)
    out_without_mask = mha(x, x, x,  mask = mask)
    
    assert not torch.allclose(out_with_mask, out_without_mask)
    

def test_mha_backward(mha_factory):
    batch_size, seq_len, d_model, num_heads = 2, 5, 16, 4
    
    x = torch.randn(batch_size, seq_len, d_model, requires_grad = True)
    
    mha = mha_factory(d_model, num_heads, .1)
    
    out = mha(x, x, x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert torch.all(x.grad != 0)