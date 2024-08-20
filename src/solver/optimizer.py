import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Dict

def build_optimizer(model: nn.Module, cfg: Dict) -> Optimizer:

    parameters = model.parameters()
    optim_type = cfg['optimizer_type']
    lr = cfg['learning_rate']

    if optim_type == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr)
    elif optim_type == 'adam':
        return torch.optim.Adam(parameters, lr=lr)
    elif optim_type == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    else:
        raise Exception('invalid optimizer, available choices adamw/adam/sgd')
    

