import pytest

from src.model.transformer import Transformer


# 1. 모델 준비
@pytest.fixture(scope = "module")
def load_model():
    model = Transformer(src_vocab_size = 1000, tgt_vocab_size = 1000)
    return model

# 2. ...