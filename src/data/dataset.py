import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
from typing import Dict
import os
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, typ: str, cfg: Dict):
        print(f"-:-:- Loading {typ} Dataset -:-:-")
        self.cfg = cfg

        self.img_data = []

        self.transform_img= transforms.Compose([
            #transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size=(self.cfg['transform']['img_height'], self.cfg['transform']['img_width'])),
            #transforms.Lambda(lambda x: x/255.0),
            transforms.ToTensor(), 
            transforms.Normalize(mean=self.cfg['transform']['mean'], std=self.cfg['transform']['std']),
             # Convert image to tensor
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(size=(self.cfg['transform']['img_height'], self.cfg['transform']['img_width'])),
            transforms.ToTensor(),  # Convert mask to tensor
        ])
        self.flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ])
        # Load image and mask paths
        img_dir = os.path.join(self.cfg[typ]['root_dir'], 'data')
        mask_dir = os.path.join(self.cfg[typ]['root_dir'], 'label')

        cls_list = os.listdir(img_dir)

        for cls in cls_list:
            img_list_pth = os.path.join(img_dir, cls)
            img_list = os.listdir(img_list_pth)
            for img_id in img_list:
                if len(img_id) < 9:
                    img_pth = os.path.join(img_list_pth, img_id)
                    if cls != 'good':
                        mask_id = img_id[:3] + '_mask.png'
                        mask_pth = os.path.join(mask_dir, cls, mask_id)
                    else:
                        mask_pth = None
                    self.img_data.append([img_pth, mask_pth])
                
    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_pth, mask_pth = self.img_data[idx]

        # Load and transform image
        img = Image.open(img_pth).convert("RGB")  # Ensure image is in RGB format
        img = self.transform_img(img)

        # Load and transform mask
        if mask_pth is not None:
            mask = Image.open(mask_pth).convert("L")  # Convert mask to grayscale (single channel)
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0).long()  # Remove channel dimension
        else:
            mask = torch.zeros((self.cfg['transform']['img_height'], self.cfg['transform']['img_width']), dtype=torch.long)

        return img, mask

def build_data_loader(typ: str, cfg: Dict) -> DataLoader:
    dataset = ImageDataset(typ, cfg)
    data_loader = DataLoader(dataset, batch_size=cfg[typ]['batch_size'], shuffle=cfg[typ]['shuffle'])
    return data_loader
