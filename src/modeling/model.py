import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, ModuleList, ReLU
from torchvision.transforms import CenterCrop
from typing import Dict

class Block(nn.Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
            
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
            
	def forward(self, x):

		return self.conv2(self.relu(self.conv1(x)))
      

class Encoder(nn.Module):
	def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
		super().__init__()
		
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
		
	def forward(self, x):

		blockOutputs = []

		for block in self.encBlocks:
			
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)

		return blockOutputs    


class Decoder(Module):
	def __init__(self, channels=(1024, 512, 256, 128, 64)):
		super().__init__()
		
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		
	def forward(self, x, encFeatures):
		
		for i in range(len(self.channels) - 1):

			x = self.upconvs[i](x)
			
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)

		return x
	
	def crop(self, encFeatures, x):
		
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)

		return encFeatures


class UNet(nn.Module):
	def __init__(self, cfg: Dict, encChannels=(3, 64, 128, 256, 512, 1024),
		decChannels=(1024, 512, 256, 128, 64),
		nbClasses=2, retainDim=True,
		outSize=(572,  572)):
		super().__init__()
		print("-:-:- Loading Model -:-:-")
		
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)

		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
		
	def forward(self, x):
		encFeatures = self.encoder(x)
		decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
		map = self.head(decFeatures)

		if self.retainDim:
			map = F.interpolate(map, self.outSize)
		return map

# class DoubleConv(nn.Module):
#   def __init__(self,in_channels,out_channels,mid_channels=None):
#     super().__init__()
#     if not mid_channels:
#       mid_channels = out_channels
#     self.double_conv = nn.Sequential(
#         nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=0,bias=False),
#         nn.BatchNorm2d(mid_channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=0,bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )

#   def forward(self,x):
#     return self.double_conv(x)

# class Down(nn.Module):
#   def __init__(self,in_channels,out_channels):
#     super().__init__()
#     self.maxpool_conv = nn.Sequential(
#         nn.MaxPool2d(2),
#         DoubleConv(in_channels,out_channels)
#     )

#   def forward(self,x):
#     return self.maxpool_conv(x)
  
# class Up(nn.Module):
#   def __init__(self,in_channels,out_channels,bilinear=True):
#     super().__init__()
#     if bilinear:
#       self.up = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
#       self.conv = DoubleConv(in_channels,out_channels,in_channels//2)
#     else:
#       self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
#       self.conv = DoubleConv(in_channels,out_channels)

#   def forward(self,x1,x2):
#     x1 = self.up(x1)

#     diffY = x2.size()[2] - x1.size()[2]
#     diffX = x2.size()[3] - x1.size()[3]

#     x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])

#     x = torch.cat([x2, x1], dim=1)
#     return self.conv(x)

# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)
    

# class UNet(nn.Module):
#     def __init__(self, cfg: Dict, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         factor = 2 if bilinear else 1
#         self.down4 = (Down(512, 1024 // factor))
#         self.up1 = (Up(1024, 512 // factor, bilinear))
#         self.up2 = (Up(512, 256 // factor, bilinear))
#         self.up3 = (Up(256, 128 // factor, bilinear))
#         self.up4 = (Up(128, 64, bilinear))
#         self.outc = (OutConv(64, n_classes))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits


def build_model(cfg: Dict) -> Module:
    model = UNet(cfg)
    return model