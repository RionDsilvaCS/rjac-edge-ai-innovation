import torch
import torch.nn as nn
from typing import Dict
from torch.nn.modules import Module

def build_loss(cfg: Dict) -> Module:

    loss_type = cfg['loss_type']

    if loss_type == 'l1_loss':
        return nn.L1Loss()
    elif loss_type == 'mse_loss':
        return nn.MSELoss()
    elif loss_type == 'cross_entropy_loss':
        return nn.CrossEntropyLoss()
    elif loss_type == 'nll_loss':
        return nn.NLLLoss()
    elif loss_type == 'bce_loss':
        return nn.BCELoss()
    elif loss_type == 'bce_with_logits_loss':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'triplet_margin_loss':
        return nn.TripletMarginLoss()
    else:
        raise Exception('invalid loss function, add the custom loss function to file')