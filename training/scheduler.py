import torch
from typing import Dict, Any


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    name = config["name"]

    if name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", 100),
            eta_min=config.get("eta_min", 1e-7),
        )
    elif name == "cosine_annealing_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 50),
            T_mult=config.get("T_mult", 2),
            eta_min=config.get("eta_min", 1e-7),
        )
    elif name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("factor", 0.5),
            patience=config.get("patience", 10),
            min_lr=config.get("min_lr", 1e-7),
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1),
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
