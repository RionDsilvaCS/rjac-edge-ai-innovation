import lightning as L
import torch
from src.config import build_config
from src.data import build_data_loader
from src.tools.train import LightningModel
from torchvision.utils import save_image
import os

cfg = build_config('exp01_config.yaml')
    
model = LightningModel.load_from_checkpoint('/home/charanarikala/rjac-edge-ai-innovation/experiments/exp03/logs/segnet_exp_01/version_5/checkpoints/epoch=49-step=850.ckpt', cfg=cfg)
model.eval()

test_loader = build_data_loader('test', cfg)

for idx, (i, o) in enumerate(test_loader, 0):
    i = i.to('cuda')
    with torch.no_grad():
        out = model(i)
    out = out.squeeze(0)
    save_id = '{}.png'.format(idx)
    save_pth = os.path.join('/home/charanarikala/rjac-edge-ai-innovation/experiments/exp03/Results/photo/' , save_id)
    save_image(tensor=out, fp=save_pth)
    print('Img saved to {}'.format(save_pth))