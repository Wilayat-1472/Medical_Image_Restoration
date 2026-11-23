"""Compatibility wrapper; downloads MedMNIST references rather than the old dataset."""
from .medmnist_data import download_medmnist


def download_low_dose_ct(save_dir, modality="ct", **kwargs):
    return download_medmnist(save_dir, modality=modality, size=kwargs.get("size", 128), limit=kwargs.get("num_samples"))
