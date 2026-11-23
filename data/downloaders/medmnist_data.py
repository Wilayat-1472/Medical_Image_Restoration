"""Public research data with provider-assigned splits, no credentials required."""
from pathlib import Path
import numpy as np
from data.io import save_image

FLAGS = {'ct': 'organamnist', 'xray': 'pneumoniamnist',
         'retinal': 'retinamnist', 'ultrasound': 'breastmnist'}


def download_medmnist(save_dir, modality='xray', size=128, limit=None):
    import medmnist
    if modality not in FLAGS:
        raise ValueError('MRI requires a local dataset; put converted references in processed/<split>/mri/clean')
    flag = FLAGS[modality]
    cache = Path(save_dir) / 'medmnist_raw'
    cache.mkdir(parents=True, exist_ok=True)
    dataset_class = getattr(medmnist, medmnist.INFO[flag]['python_class'])
    count = 0
    for split in ('train', 'val', 'test'):
        dataset = dataset_class(split=split, root=str(cache), download=True, size=size)
        for i in range(len(dataset) if limit is None else min(limit, len(dataset))):
            image, _label = dataset[i]
            save_image(Path(save_dir) / 'processed' / split / modality / 'clean' / f'{flag}_{i:06d}.png', np.asarray(image))
            count += 1
    return count
