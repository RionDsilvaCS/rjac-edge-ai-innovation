import lightning as L
import torch
from src.config import build_config
from src.data import build_data_loader
from src.tools.train import LightningModel
from torchvision.utils import save_image

cfg = build_config('exp02_deeplabv3_mobilenet.yaml')
    
model = LightningModel.load_from_checkpoint('./data/model_chckpt/deeplab/epoch=79-train/loss=0.0071.ckpt', cfg=cfg)
model.eval()

test_loader = build_data_loader('test', cfg)

for idx, (i, o) in enumerate(test_loader, 0):
    i = i.to('cuda')
    with torch.no_grad():
        out = model(i)
    out = out['out'].squeeze(0)
    save_id = '{}.png'.format(idx)
    save_pth = './experiments/exp02/results/ver_0/img/' + save_id
    save_image(tensor=out, fp=save_pth)
    print('Img saved to {}'.format(save_pth))
