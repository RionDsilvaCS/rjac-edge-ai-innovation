import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms 
from typing import List, Dict
import os
import numpy as np
import random

ALL_CLASSES = ['background', 'defect']

LABEL_COLORS_LIST = [
    (0, 0, 0),
    (255, 255, 255),
]

def set_class_values(all_classes, classes_to_train):
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    mask = mask.numpy()
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)

    return label_mask

class ImageDataset(Dataset):
    def __init__(self, typ: str, cfg: Dict):
        print("-:-:- Loading {} Dataset -:-:-".format(typ))
        self.cfg = cfg

        self.img_data = []

        self.transform_img= transforms.Compose([
            transforms.Resize(size=(self.cfg['transform']['img_height'], self.cfg['transform']['img_width'])),
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize(mean=self.cfg['transform']['mean'], std=self.cfg['transform']['std']),
        ])

        self.transform_mask= transforms.Compose([
            transforms.Resize(size=(self.cfg['transform']['msk_height'], self.cfg['transform']['msk_width'])),
        ])

        self.flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
        ])

        self.class_values = set_class_values(
            ALL_CLASSES, ALL_CLASSES
        )

        img_dir = os.path.join(self.cfg[typ]['root_dir'], 'data')
        mask_dir = os.path.join(self.cfg[typ]['root_dir'], 'label')

        cls_list = os.listdir(img_dir)

        for cls in cls_list:
            img_list_pth = os.path.join(img_dir, cls)
            img_list = os.listdir(img_list_pth)
            for img_id in img_list:
                if len(img_id) < 12:
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
        num = random.choice([0, 1])

        img = io.read_image(img_pth)
        if mask_pth is not None:
            mask = io.read_image(mask_pth)
            mask = self.transform_mask(mask)
            mask[mask < 200] = 0
            mask[mask >= 200] = 255
            mask = mask.permute(1, 2, 0)
        else:
            mask = torch.zeros((self.cfg['transform']['msk_height'], self.cfg['transform']['msk_width'], 3), dtype=torch.long)

        mask = get_label_mask(mask, self.class_values, LABEL_COLORS_LIST)
        
        img = self.transform_img(img)
        

        mask = torch.tensor(mask, dtype=torch.long)

        if num == 1:
            img = self.flip(img)
            mask = self.flip(mask)

        return img.float(), mask
    

def build_data_loader(typ: str, cfg: Dict) -> DataLoader:
    dataset = ImageDataset(typ, cfg)
    data_loader = DataLoader(dataset, cfg[typ]['batch_size'], cfg[typ]['shuffle'], num_workers=15)

    return data_loader

    