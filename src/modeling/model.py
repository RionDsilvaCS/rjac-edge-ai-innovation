import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from typing import Dict, Tuple

class CustomFCN(nn.Module):
    def __init__(self, cfg: Dict,
                 nb_classes: int = 2,  
                 retain_dim: bool = True,
                 out_size: Tuple[int, int] = (520,520)):
        super(CustomFCN, self).__init__()
        print("-:-:- Loading Model -:-:-")

        # Initialize a pre-trained FCN model
        self.fcn = segmentation.fcn_resnet50(pretrained=True)
        
        # Replace the classifier with a new one for the desired number of output classes
        in_channels = self.fcn.classifier[0].in_channels
        self.fcn.classifier = nn.Sequential(
            nn.Conv2d(in_channels, nb_classes, kernel_size=1),
            nn.BatchNorm2d(nb_classes)
        )

        # Store parameters for resizing output
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the model
        features = self.fcn(x)
        x = features['out']

        if self.retain_dim:
            x = torch.nn.functional.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        print(x.shape)
        return x

def build_model(cfg: Dict) -> nn.Module:
    model = CustomFCN(cfg)
    return model
