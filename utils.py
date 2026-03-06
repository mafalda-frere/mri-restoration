import torch


def get_params(model: torch.nn.Module):
    n_params = 0
    for param in model.parameters(): 
        n_params += param.shape.numel()
    return n_params
