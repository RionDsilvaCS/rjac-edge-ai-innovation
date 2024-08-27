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
        self.save_hyperparameters(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return build_optimizer(self.model, self.cfg)
