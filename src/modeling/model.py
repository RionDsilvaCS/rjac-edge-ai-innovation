import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
import torchvision.models.segmentation as seg
from torch.nn import Conv2d
from typing import Dict

class DeepLabv3_Resnet50(nn.Module):
	def __init__(self, cfg: Dict):
		super().__init__()
		print("-:-:- Loading Model -:-:-")
		
		self.deeplab = seg.deeplabv3_resnet50(weights=seg.DeepLabV3_ResNet50_Weights.DEFAULT)
		self.deeplab.aux_classifier[4] = Conv2d(256, cfg['num_classes'], kernel_size=(1, 1), stride=(1, 1))
		self.deeplab.classifier[4] = Conv2d(256, cfg['num_classes'], kernel_size=(1, 1), stride=(1, 1))

	def forward(self, x):
		x = self.deeplab(x)
		return x

def build_model(cfg: Dict) -> Module:
    model = DeepLabv3_Resnet50(cfg)
    return model