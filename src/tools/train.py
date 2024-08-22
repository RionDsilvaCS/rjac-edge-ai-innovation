import lightning as L
import torch
from src.solver import build_optimizer, build_loss
from src.modeling import build_model

class LightningModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = build_model(self.cfg)
        self.criterion = build_loss(self.cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out['out'], y)
        # loss_2 = self.criterion(out['aux'], y)
        # loss = loss_1 + loss_2

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalar("train/loss",
                                            loss,
                                            batch_idx)
         
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out['out'], y)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalar("val/loss",
                                            loss,
                                            batch_idx)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out['out'], y)
        
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalar("test/loss",
                                            loss,
                                            batch_idx)
        
        return loss

    def configure_optimizers(self):
        return build_optimizer(self.model, self.cfg)
