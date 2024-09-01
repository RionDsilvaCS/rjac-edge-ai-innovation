import torch
import torch.nn as nn
from typing import Dict

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x, indices = self.pool(x)
        return x, indices

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class SegNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SegNet, self).__init__()
        
        # Encoder
        self.encoders = nn.ModuleList([
            EncoderBlock(num_channels, 32),
            EncoderBlock(32, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256)
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1)  # Reduce channels to 256
        )
        
        # Decoder
        self.decoders = nn.ModuleList([
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 32)
        ])
        
        # Final layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        indices_list = []
        sizes = []  # List to keep track of sizes for unpooling
        
        # Encoding
        for encoder in self.encoders:
            sizes.append(x.size())  # Keep track of the size before max pooling
            x, indices = encoder(x)
            skip_connections.append(x)
            indices_list.append(indices)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoding
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, indices_list[-(i+1)], sizes[-(i+1)])
        
        # Final classification
        x = self.final_conv(x)
        return x

def build_model(cfg: Dict) -> nn.Module:
    return SegNet(num_channels=cfg['num_channels'], num_classes=cfg['num_classes'])

# Example usage
cfg = {'num_channels': 3, 'num_classes': 10}
model = build_model(cfg)