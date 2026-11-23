from typing import Dict, Any, Optional, Union

import torch
import pytorch_lightning as pl

from models import LightUNet, RestormerUNet
from .losses import CombinedLoss
from .scheduler import get_scheduler


from models.factory import create_model


class MedicalRestorationModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = create_model(config)
        self.criterion = CombinedLoss(
            l1_weight=config["loss"]["l1_weight"],
            ssim_weight=config["loss"]["ssim_weight"],
            perceptual_weight=config["loss"]["perceptual_weight"],
            charbonnier_weight=config["loss"]["charbonnier_weight"],
            edge_weight=config["loss"].get("edge_weight", 0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        degraded = batch["degraded"]
        clean = batch["clean"]
        restored = self(degraded)
        losses = self.criterion(restored, clean)
        self.log("train_loss", losses["total"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_l1", losses["l1"], on_step=False, on_epoch=True)
        self.log("train_ssim", 1 - losses["ssim"], on_step=False, on_epoch=True, prog_bar=True)
        return losses["total"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        degraded = batch["degraded"]
        clean = batch["clean"]
        restored = self(degraded)
        losses = self.criterion(restored, clean)
        self.log("val_loss", losses["total"], on_epoch=True, prog_bar=True)
        self.log("val_l1", losses["l1"], on_epoch=True)
        self.log("val_ssim", 1 - losses["ssim"], on_epoch=True, prog_bar=True)
        return {"restored": restored, "clean": clean, "degraded": degraded}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> dict:
        opt_cfg = self.config["training"]["optimizer"]
        sched_cfg = self.config["training"]["scheduler"]

        if opt_cfg["name"] == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=opt_cfg["lr"],
                weight_decay=opt_cfg.get("weight_decay", 0),
            )
        elif opt_cfg["name"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=opt_cfg["lr"],
                weight_decay=opt_cfg.get("weight_decay", 0.01),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg['name']}")

        scheduler = get_scheduler(optimizer, sched_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
