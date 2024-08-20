import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from src.config import build_config
from src.data import build_data_loader
from src.tools.train import LightningModel

torch.set_float32_matmul_precision('high')
torch.manual_seed(314159)

cfg = build_config('exp01_config.yaml')
    
model = LightningModel(cfg)
# model = LightningModel.load_from_checkpoint('pth/name.ckpt', config=config)

train_loader = build_data_loader('train', cfg)
val_loader = build_data_loader('val', cfg)
test_loader = build_data_loader('test', cfg)

logger = TensorBoardLogger(cfg['log_path'], name=cfg['log_name'])

trainer = L.Trainer(accelerator=cfg['accelerator'], 
                    devices=cfg['devices'],

                    min_epochs=cfg['min_epochs'],
                    max_epochs=cfg['max_epochs'],
                    log_every_n_steps=cfg['log_every_n_steps'],
                    # check_val_every_n_epoch=cfg['check_val_every_n_epoch'],

                    enable_checkpointing=True,
                    # callbacks=[cfg['checkpoint_callback']],
                    
                    limit_train_batches=cfg['limit_train_batches'],
                    limit_val_batches=cfg['limit_val_batches'],
                    limit_test_batches=cfg['limit_test_batches'],
                    
                    logger=logger
                    )

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model=model, dataloaders=test_loader)