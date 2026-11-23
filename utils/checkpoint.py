"""Portable, strict PyTorch checkpoints with embedded architecture config."""
from pathlib import Path
import torch
from models.factory import create_model


def save_checkpoint(path, model, config, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": config,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                **metadata}, path)


def load_model(path, config=None, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in payload:
        config = payload.get("config", payload.get("hyper_parameters", {}).get("config", config))
        weights = payload["state_dict"]
        if any(k.startswith("model.") for k in weights):
            weights = {k.removeprefix("model."): v for k, v in weights.items() if k.startswith("model.")}
    else:
        weights = payload
    if config is None:
        raise ValueError("A config is required for a legacy weights-only checkpoint")
    model = create_model(config)
    model.load_state_dict(weights, strict=True)
    return model.eval().to(device), config
