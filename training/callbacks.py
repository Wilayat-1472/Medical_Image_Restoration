import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ImageLoggingCallback(Callback):
    def __init__(self, log_every_n_epochs: int = 10, num_samples: int = 4):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        batch = next(iter(trainer.datamodule.val_dataloader())) if getattr(trainer, "datamodule", None) is not None else None
        if batch is None:
            return

        degraded = batch["degraded"][: self.num_samples].to(pl_module.device)
        clean = batch["clean"][: self.num_samples].to(pl_module.device)

        pl_module.eval()
        with torch.no_grad():
            restored = pl_module(degraded)
        pl_module.train()

        images = {
            "degraded": degraded.cpu(),
            "restored": restored.cpu(),
            "clean": clean.cpu(),
        }

        for name, tensor in images.items():
            tensor = tensor * 0.5 + 0.5
            grid = torch.cat([tensor[i] for i in range(min(self.num_samples, tensor.shape[0]))], dim=2)
            trainer.logger.experiment.add_image(
                f"val/{name}",
                grid,
                trainer.current_epoch,
            )
