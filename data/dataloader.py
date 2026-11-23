from torch.utils.data import DataLoader
from .dataset import MedicalRestorationDataset
from .preprocessing import MedicalTransform


def _loader(root_dir, split, batch_size=4, num_workers=2, pin_memory=True,
            target_size=256, seed=42, train_split=0.8, val_split=0.1,
            degradation_config=None, identity_probability=0.0, **kwargs):
    dataset = MedicalRestorationDataset(
        root_dir, transform=MedicalTransform(target_size=(target_size, target_size), augment=split == 'train'),
        split=split, seed=seed, train_split=train_split, val_split=val_split,
        degradation_config=degradation_config, identity_probability=identity_probability)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=pin_memory, shuffle=split == 'train', drop_last=False)


def create_train_loader(root_dir, **kwargs):
    return _loader(root_dir, 'train', **kwargs)


def create_val_loader(root_dir, **kwargs):
    return _loader(root_dir, 'val', **kwargs)


def create_test_loader(root_dir, **kwargs):
    return _loader(root_dir, 'test', **kwargs)
