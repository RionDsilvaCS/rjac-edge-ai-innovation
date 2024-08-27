import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from typing import Dict, Tuple

class CustomFCN(nn.Module):
    def __init__(self, cfg: Dict,
                 nb_classes: int = 2,
                 retain_dim: bool = True,
                 out_size: Tuple[int, int] = (520, 520)):
        super(CustomFCN, self).__init__()
        print("-:-:- Loading Model -:-:-")

        # Initialize a pre-trained FCN model
        self.fcn = segmentation.fcn_resnet50(pretrained=True)

        # Modify the first convolution layer to reduce channels
        self.fcn.backbone.conv1 = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fcn.backbone.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fcn.backbone.relu = nn.ReLU(inplace=True)
        self.fcn.backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # Reduced layers for simplicity
        self.fcn.backbone.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.fcn.backbone.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.fcn.backbone.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.fcn.backbone.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.fcn.classifier = nn.Sequential(
            nn.Conv2d(1024, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.fcn.aux_classifier = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(128, 21, kernel_size=(1, 1), stride=(1, 1))
        )

        #self.fcn.backbone.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)


        
        
        
        


        #         #Replace the classifier with a new one for the desired number of output classes
        in_channels = self.fcn.classifier[0].in_channels
        self.fcn.classifier = nn.Sequential(
        nn.Conv2d(in_channels, nb_classes, kernel_size=1),
        nn.BatchNorm2d(nb_classes)
         )

        # # Store parameters for resizing output
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
    print(model)
    return model