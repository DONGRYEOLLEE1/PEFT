import torch
import torch.nn as nn


class IA3SelfAttention(nn.Module):
    """IA3 적용 SA-layer"""
    def __init__(self, original_attention):
        super().__init__()
        self.original_attention = original_attention
        hidden_size = self.original_attention.key.in_features
        
        # IA3 학습 파라미터 (key , value scaling)
        self.l_key = nn.Parameter(torch.ones(hidden_size))
        self.l_value = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, hidden_states):
        query = self.original_attention.query(hidden_states)
        key = self.original_attention.key(hidden_states)
        value = self.original_attention.value(hidden_states)
        
        key = key * self.l_key
        value = value * self.l_value
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_probs = nn.functional.softmax(attention_scores, dim = -1)
        context = torch.matmul(attention_probs, value)
        
        return context
    
    
class IA3FFN(nn.Module):
    """IA3 적용 FFN layer"""
    def __init__(self, original_ffn):
        super().__init__()
        self.original_ffn = original_ffn
        intermediate_size = original_ffn.dense.out_features
        
        # IA3 학습 파라미터 
        self.l_ffn = nn.Parameter(torch.ones(intermediate_size))
        
    def forward(self, hidden_states):
        hidden_states = self.original_ffn.dense(hidden_states)
        hidden_states = self.original_ffn.intermediate_act_fn(hidden_states)
        
        hidden_states = hidden_states * self.l_ffn
        
        return hidden_states
    
    
def apply_ia3(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for layer in model.bert.encoder.layer:
        layer.attention.self = IA3SelfAttention(layer.attention.self)
        layer.intermediate = IA3FFN(layer.intermediate)
        
    return model