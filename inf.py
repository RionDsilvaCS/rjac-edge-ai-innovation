import lightning as L
import torch
from src.config import build_config
from src.data import build_data_loader
from src.tools.train import LightningModel
from torchvision.utils import save_image
import os

cfg = build_config('exp01_config.yaml')
    
model = LightningModel.load_from_checkpoint(r'E:\edge AI\rjac-edge-ai-innovation\experiments\exp02\logs\fcn_exp_02\version_18\checkpoints\epoch=49-step=850.ckpt', cfg=cfg)
model.eval()

test_loader = build_data_loader('test', cfg)

for idx, (i, o) in enumerate(test_loader, 0):
    i = i.to('cpu')
    with torch.no_grad():
        out = model(i)
    out = out.squeeze(0)
    save_id = '{}.png'.format(idx)
    save_pth = os.path.join(r'E:\edge AI\rjac-edge-ai-innovation\experiments\exp02\Results\img' , save_id)
    save_image(tensor=out, fp=save_pth)
    print('Img saved to {}'.format(save_pth))