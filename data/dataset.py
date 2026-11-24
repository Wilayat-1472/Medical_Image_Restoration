"""Aligned restoration pairs with reproducible, disjoint train/val/test splits."""
import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .io import load_image, RASTER_EXTENSIONS
from .degradation import apply_mixed_degradation
from .preprocessing import MedicalTransform


def _find_images(root_dir):
    return sorted(str(p) for p in Path(root_dir).rglob('*')
                  if p.is_file() and p.suffix.lower() in RASTER_EXTENSIONS)


def discover_pairs(root_dir):
    root = Path(root_dir)
    clean_dirs = sorted(p for p in root.rglob('clean') if p.is_dir())
    if root.name == 'clean':
        clean_dirs = [root]
    pairs = []
    for clean_dir in clean_dirs:
        for path in _find_images(clean_dir):
            counterpart = clean_dir.parent / 'degraded' / Path(path).relative_to(clean_dir)
            pairs.append((path, str(counterpart) if counterpart.is_file() else None))
    if not clean_dirs:
        pairs = [(p, None) for p in _find_images(root)
                 if 'degraded' not in Path(p).relative_to(root).parts]
    return pairs


class MedicalRestorationDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_split=0.8,
                 val_split=0.1, seed=42, max_degradations=3, degradation_config=None,
                 image_paths=None, identity_probability=0.0):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown split: {split}')
        if not 0 < train_split < 1 or not 0 < val_split < 1 - train_split:
            raise ValueError('Train, validation and test fractions must be positive and sum to 1')
        self.root_dir, self.split, self.seed = str(root_dir), split, seed
        self.transform = transform or MedicalTransform(augment=split == 'train')
        if not 0 <= identity_probability <= 1:
            raise ValueError("identity_probability must be between 0 and 1")
        self.identity_probability = identity_probability
        self.max_degradations = max_degradations
        self.degradation_config = degradation_config
        root = Path(root_dir)
        # Official, preassigned splits (e.g. MedMNIST) take precedence.
        explicit = [(root / name).is_dir() for name in ('train', 'val', 'test')] if image_paths is None else [False] * 3
        if any(explicit) and not all(explicit):
            raise ValueError('Explicit data splits require train/, val/ and test/ directories')
        if all(explicit):
            self.pairs = discover_pairs(root / split)
        else:
            pairs = discover_pairs(root) if image_paths is None else [(str(p), None) for p in sorted(image_paths)]
            random.Random(seed).shuffle(pairs)
            n = len(pairs)
            n_train, n_val = int(n * train_split), int(n * val_split)
            bounds = {'train': (0, n_train), 'val': (n_train, n_train + n_val), 'test': (n_train + n_val, n)}
            start, end = bounds[split]
            self.pairs = pairs[start:end]
        if not self.pairs:
            raise ValueError(f'No images in {split} split at {root_dir}; prepare more data or check its layout')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, degraded_path = self.pairs[idx]
        clean = load_image(clean_path)
        if self.split == 'train':
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
        else:
            seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{Path(clean_path).relative_to(self.root_dir)}'.encode()).digest()[:4], 'little')
        if degraded_path:
            degraded = load_image(degraded_path)
            if degraded.shape != clean.shape:
                raise ValueError(f'Pair dimensions differ: {clean_path} and {degraded_path}')
            degradations = ['paired']
        elif self.split == "train" and random.Random(seed).random() < self.identity_probability:
            degraded, degradations = clean.copy(), ["identity"]
        else:
            result = apply_mixed_degradation(clean, self.max_degradations,
                                             rng=random.Random(seed), np_rng=np.random.default_rng(seed),
                                             config=self.degradation_config)
            degraded, degradations = result['degraded'], result['degradations']
        # Reset the same local RNG for both images so geometry stays aligned.
        clean_tensor = self.transform(clean, rng=np.random.default_rng(seed))
        degraded_tensor = self.transform(degraded, rng=np.random.default_rng(seed))
        return {'degraded': degraded_tensor, 'clean': clean_tensor,
                'clean_path': clean_path, 'degraded_path': degraded_path or clean_path,
                'degradations': ';'.join(degradations)}


class SyntheticDegradationDataset(MedicalRestorationDataset):
    def __init__(self, image_paths, transform=None, max_degradations=3, split='train', **kwargs):
        # Paths may live outside the process working directory. Use their common
        # parent for portable validation seeds, not relative_to('.') in __getitem__.
        paths = [Path(p).expanduser().resolve() for p in image_paths]
        if not paths:
            raise ValueError('Provide at least one reference image path')
        root = os.path.commonpath([str(p.parent) for p in paths])
        super().__init__(root, image_paths=paths, transform=transform, split=split,
                         max_degradations=max_degradations, **kwargs)


class UnlabeledMedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = _find_images(root_dir)
        self.transform = transform or MedicalTransform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        return {'image': self.transform(load_image(path)), 'path': path}
