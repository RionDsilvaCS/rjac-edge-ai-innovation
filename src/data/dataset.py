import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms 
from typing import Dict
import random
import numpy as np
import os

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

def normalize_image(x):
    return x / 255.0

class ImageDataset(Dataset):
    def __init__(self, typ: str, cfg: Dict):
        print("-:-:- Loading {} Dataset -:-:-".format(typ))
        self.cfg = cfg

        self.img_data = []

        self.transform_img= transforms.Compose([
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size=(self.cfg['transform']['img_height'], self.cfg['transform']['img_width'])),
            transforms.Lambda(normalize_image),
            transforms.Normalize(mean=self.cfg['transform']['mean'], std=self.cfg['transform']['std']),
        ])

        self.transform_mask= transforms.Compose([
            transforms.Resize(size=(self.cfg['transform']['img_height'], self.cfg['transform']['img_width'])),
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
        num = random.choice([0, 1])

        img = io.read_image(img_pth)
        if mask_pth is not None:
            mask = io.read_image(mask_pth)
            mask = self.transform_mask(mask)
            mask[mask < 200] = 0
            mask[mask >= 200] = 255
            mask = mask.permute(1, 2, 0)
            # mask = mask.squeeze(0)
            
        else:
            mask = torch.zeros((self.cfg['transform']['img_height'], self.cfg['transform']['img_width'], 3), dtype=torch.long)

        mask = get_label_mask(mask, self.class_values, LABEL_COLORS_LIST)

        
        img = self.transform_img(img)
        
        mask = torch.tensor(mask, dtype=torch.long)

        if num == 1:
            img = self.flip(img)
            mask = self.flip(mask)
        # print("data", img.shape, mask.shape)
        return img.float(), mask.long()
    

def build_data_loader(typ: str, cfg: Dict) -> DataLoader:
    dataset = ImageDataset(typ, cfg)
    data_loader = DataLoader(dataset, cfg[typ]['batch_size'], cfg[typ]['shuffle'])

    return data_loader