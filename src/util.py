import torch

def print_trainable_params(model: torch.nn.Module, model_name: str):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"=== {model_name} ===")
    print(f"학습 가능 파라미터: {format(trainable_params, ",")}")
    print(f"전체 파라미터: {format(total_params, ",")}")
    print(f"학습 비율: {100 * (trainable_params / total_params)}%")