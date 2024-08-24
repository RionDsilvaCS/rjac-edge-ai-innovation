import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms 
from torchvision.transforms.functional import rgb_to_grayscale
from typing import List, Dict
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

class ImageDataset(Dataset):
    def __init__(self, typ: str, cfg: Dict):
        print("-:-:- Loading {} Dataset -:-:-".format(typ))
        self.cfg = cfg

        self.img_data = []

        self.transform = transforms.Compose([
            transforms.Resize(size=(self.cfg['transform']['img_height'], self.cfg['transform']['img_width'])),
        ])

        self.class_values = set_class_values(
            ALL_CLASSES, ALL_CLASSES
        )

        img_dir = os.path.join(self.cfg[typ]['root_dir'], 'data')
        seg_mask_dir = os.path.join(self.cfg[typ]['root_dir'], 'label')
        edg_mask_dir = os.path.join(self.cfg[typ]['root_dir'], 'edge')

        cls_list = os.listdir(img_dir)

        for cls in cls_list:
            img_list_pth = os.path.join(img_dir, cls)
            img_list = os.listdir(img_list_pth)
            for img_id in img_list:
                if len(img_id) < 14:
                    img_pth = os.path.join(img_list_pth, img_id)
                    if cls != 'good':
                        mask_id = img_id[:3] + '_mask.png'
                        seg_mask_pth = os.path.join(seg_mask_dir, cls, mask_id)
                        edg_mask_pth = os.path.join(edg_mask_dir, cls, mask_id)
                    else:
                        seg_mask_pth, edg_mask_pth = None, None
                    self.img_data.append([img_pth, seg_mask_pth, edg_mask_pth])
                
    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_pth, seg_mask_pth, edg_mask_pth = self.img_data[idx]
        img = io.read_image(img_pth)

        if seg_mask_pth is not None:
            seg_mask = io.read_image(seg_mask_pth)
            seg_mask = self.transform(seg_mask)
            seg_mask[seg_mask < 200] = 0
            seg_mask[seg_mask >= 200] = 255
            seg_mask = seg_mask.permute(1, 2, 0)

            edge_mask = io.read_image(edg_mask_pth)
            edge_mask = self.transform(edge_mask)
            edge_mask = rgb_to_grayscale(edge_mask)
            
        else:
            seg_mask = torch.zeros((self.cfg['transform']['img_height'], self.cfg['transform']['img_width'], 3), dtype=torch.long)
            edge_mask = torch.zeros((1, self.cfg['transform']['img_height'], self.cfg['transform']['img_width']), dtype=torch.float32)

        seg_mask = get_label_mask(seg_mask, self.class_values, LABEL_COLORS_LIST)
        seg_mask = torch.tensor(seg_mask, dtype=torch.long)
        
        img = self.transform(img)

        return img.float(), seg_mask, edge_mask.float()
    

def build_data_loader(typ: str, cfg: Dict) -> DataLoader:
    dataset = ImageDataset(typ, cfg)
    data_loader = DataLoader(dataset, cfg[typ]['batch_size'], cfg[typ]['shuffle'], num_workers=15)

    return data_loader

    