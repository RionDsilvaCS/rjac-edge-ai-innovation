import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
# import torchvision.models.segmentation as seg
import torchvision.models.detection as seg
from torch.nn import Conv2d
from typing import Dict

class maskrcnn_Resnet50_fpn_v2(nn.Module):
	def __init__(self, cfg: Dict):
		super().__init__()
		print("-:-:- Loading Model -:-:-")
		
		self.mask_rcnn = seg.maskrcnn_resnet50_fpn_v2(weights=seg.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
		# self.deeplab.aux_classifier[4] = Conv2d(256, cfg['num_classes'], kernel_size=(1, 1), stride=(1, 1))
		# self.deeplab.classifier[4] = Conv2d(256, cfg['num_classes'], kernel_size=(1, 1), stride=(1, 1))


	def forward(self, x):
		x = self.mask_rcnn(x)
		return x

def build_model(cfg: Dict) -> Module:

	model = maskrcnn_Resnet50_fpn_v2(cfg)
	model.mask_rcnn.roi_heads = nn.Sequential(*list(model.mask_rcnn.roi_heads.children())[-3:])
	cls = list(model.mask_rcnn.roi_heads.children())[-1][-1]
	cls = Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
	list(model.mask_rcnn.roi_heads.children())[-1][-1] = cls
	return model

	

	# return model

cfg = build_model('exp01_config.yaml')
print(cfg)