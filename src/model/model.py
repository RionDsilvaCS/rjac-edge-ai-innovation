import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from typing import Dict
from .hrnet.model import HighResolutionNet

def build_model(cfg: Dict) -> Module:
    model = HighResolutionNet(cfg)
    print('-:-:- Loading Model -:-:-')
    return model