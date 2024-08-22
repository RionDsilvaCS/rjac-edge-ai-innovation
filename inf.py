import lightning as L
import torch
from src.config import build_config
from src.data import build_data_loader
from src.tools.train import LightningModel
from torchvision.utils import save_image

cfg = build_config('exp01_deeplabv3_resnet50.yaml')
    
model = LightningModel.load_from_checkpoint('./experiments/exp01/logs/exp_01_deeplabv3_resnet50/version_0/checkpoints/epoch=199-step=6800.ckpt', cfg=cfg)
model.eval()

test_loader = build_data_loader('test', cfg)

for idx, (i, o) in enumerate(test_loader, 0):
    i = i.to('cuda')
    with torch.no_grad():
        out = model(i)
    out = out['out'].squeeze(0)
    save_id = '{}.png'.format(idx)
    save_pth = './experiments/exp01/results/img/' + save_id
    save_image(tensor=out, fp=save_pth)
    print('Img saved to {}'.format(save_pth))
