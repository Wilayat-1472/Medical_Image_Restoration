from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import sys
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train medical image restoration model")
    parser.add_argument("--config", default="configs/cpu_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", default="./datasets/processed", help="Path to processed data")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--checkpoint", default=None, help="Resume a Lightning .ckpt including optimizer state")
    initialization.add_argument("--init_weights", default=None, help="Initialize matching model weights from a portable .pt for fine-tuning")
    parser.add_argument("--model_name", default=None, help="Override model name")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--output_dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(config["project"].get("seed", 42), workers=True)

    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["max_epochs"] = args.epochs
    if args.lr:
        config["training"]["optimizer"]["lr"] = args.lr

    config["data"]["root_dir"] = args.data_dir

    from data.dataloader import create_train_loader, create_val_loader
    from training.trainer import MedicalRestorationModule

    train_loader = create_train_loader(
        root_dir=args.data_dir,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        target_size=config["data"]["target_size"],
        seed=config["project"].get("seed", 42),
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.1),
        degradation_config=config["data"].get("degradation"),
        identity_probability=config["data"].get("identity_probability", 0.0),
    )

    val_loader = create_val_loader(
        root_dir=args.data_dir,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        target_size=config["data"]["target_size"],
        seed=config["project"].get("seed", 42),
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.1),
        degradation_config=config["data"].get("degradation"),
        identity_probability=config["data"].get("identity_probability", 0.0),
    )

    model = MedicalRestorationModule(config)
    if args.init_weights:
        from utils.checkpoint import load_model
        pretrained, _ = load_model(args.init_weights, device="cpu")
        model.model.load_state_dict(pretrained.state_dict(), strict=True)

    accelerator = config["training"].get("accelerator", "auto")
    if accelerator == "cpu":
        torch.set_num_threads(config["training"].get("num_threads", 2))

    logger = TensorBoardLogger("logs", name=config["project"]["name"])
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if accelerator == "cpu":
        early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
        callbacks.append(early_stop)

    limits = {}
    for key in ("limit_train_batches", "limit_val_batches"):
        if getattr(args, key) is not None:
            limits[key] = getattr(args, key)
    trainer = pl.Trainer(
        **limits,
        accelerator=accelerator,
        devices=config["training"].get("devices", 1),
        max_epochs=config["training"]["max_epochs"],
        precision=config["training"].get("precision", "32"),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config["training"].get("log_every_n_steps", 50),
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)

    # Export the best validation checkpoint, not the last epoch's weights.
    best = trainer.checkpoint_callback.best_model_path
    if not best:
        raise RuntimeError("No validation checkpoint was saved")
    model = MedicalRestorationModule.load_from_checkpoint(best, config=config)
    from utils.checkpoint import save_checkpoint
    best_path = str(Path(args.output_dir) / f"{config['model']['name']}_best.pt")
    score = trainer.checkpoint_callback.best_model_score
    save_checkpoint(best_path, model.model, config,
                    validation_loss=float(score), source_checkpoint=Path(best).name)
    print(f"Best validation model exported to {best_path}")



if __name__ == "__main__":
    main()
