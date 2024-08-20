import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms 
from typing import List, Dict
import os

class ImageDataset(Dataset):
    def __init__(self, typ: str, cfg: Dict):
        print("-:-:- Loading {} Dataset -:-:-".format(typ))
        self.cfg = cfg

        self.img_data = []

        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size=(self.cfg['transform']['img_height'], self.cfg['transform']['img_width'])),
        ])

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
        img = io.read_image(img_pth)
        if mask_pth is not None:
            mask = io.read_image(mask_pth)
            mask = self.transform(mask)
        else:
            mask = torch.zeros((1, 224, 224), dtype=torch.float32)

        img = self.transform(img)
        
        return img, mask
    

def build_data_loader(typ: str, cfg: Dict) -> DataLoader:
    dataset = ImageDataset(typ, cfg)
    data_loader = DataLoader(dataset, cfg[typ]['batch_size'], cfg[typ]['shuffle'], num_workers=15)

    return data_loader

    