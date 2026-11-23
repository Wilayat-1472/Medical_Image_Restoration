"""Model construction without importing the training framework or VGG."""
from .light_unet import LightUNet
from .restormer_unet import RestormerUNet


def create_model(config):
    cfg = config["model"]
    common = dict(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"])
    if cfg["name"] == "light_unet":
        params = {k: v for k, v in cfg["light_unet"].items() if k in {"base_filters", "depth", "use_se"}}
        return LightUNet(**common, **params)
    if cfg["name"] == "restormer_unet":
        return RestormerUNet(**common, **cfg["restormer_unet"])
    raise ValueError(f"Unknown model: {cfg['name']}")
