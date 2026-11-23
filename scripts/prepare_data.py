from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import numpy as np
import cv2
from data.io import load_image, save_image, RASTER_EXTENSIONS


def normalized_slice(image):
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError('Non-finite medical image')
    low, high = np.percentile(image, [0.5, 99.5])
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def read_slices(path, slice_axis=0):
    name = path.name.lower()
    if path.suffix.lower() in RASTER_EXTENSIONS:
        yield load_image(path)
        return
    if name.endswith('.dcm'):
        import pydicom
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
        ds = pydicom.dcmread(path)
        if getattr(ds, 'SamplesPerPixel', 1) > 1:
            from data.io import unit_float
            yield unit_float(ds.pixel_array)
            return
        pixels = apply_voi_lut(apply_modality_lut(ds.pixel_array, ds), ds)
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            pixels = pixels.max() - pixels
        array = pixels
    elif name.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        array = nib.load(path).get_fdata(dtype=np.float32)
    elif name.endswith('.npy'):
        array = np.load(path, allow_pickle=False)
    elif name.endswith('.h5'):
        import h5py
        with h5py.File(path, 'r') as handle:
            if 'reconstruction_rss' in handle:
                array = handle['reconstruction_rss'][:]
            elif 'reconstruction_esc' in handle:
                array = handle['reconstruction_esc'][:]
            elif 'kspace' in handle:
                kspace = handle['kspace'][:]
                image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))
                array = np.sqrt(np.sum(np.abs(image)**2, axis=1)) if image.ndim == 4 else np.abs(image)
            else:
                raise ValueError('HDF5 requires reconstruction_rss, reconstruction_esc, or kspace')
    else:
        raise ValueError(f'Unsupported file: {path}')
    if array.ndim == 2:
        yield normalized_slice(array)
    elif array.ndim == 3:
        for image in np.moveaxis(array, slice_axis, 0):
            yield normalized_slice(image)
    else:
        raise ValueError(f'Expected a 2D image or 3D volume, got {array.shape}')


def prepare(input_dir, output_dir, target_size=256, modality='all', slice_axis=0):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    modalities = ['ct', 'mri', 'xray', 'retinal', 'ultrasound'] if modality == 'all' else [modality]
    count, failures = 0, []
    for mod in modalities:
        root = input_dir / mod / 'images'
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file() or not path.name.lower().endswith(tuple(RASTER_EXTENSIONS | {'.dcm', '.nii', '.nii.gz', '.npy', '.h5'})):
                continue
            # Keep slices from one volume together; never split a volume across datasets.
            relative = str(path.relative_to(root))
            digest = hashlib.sha256(f'{mod}/{relative}'.encode()).hexdigest()
            bucket = int(digest[:8], 16) % 100
            split = 'train' if bucket < 80 else 'val' if bucket < 90 else 'test'
            try:
                for i, image in enumerate(read_slices(path, slice_axis)):
                    image = cv2.resize(image, (target_size, target_size))
                    save_image(output_dir / split / mod / 'clean' / f'{digest[:16]}_{i:05d}.png', image)
                    count += 1
            except Exception as exc:
                failures.append(f'{path}: {exc}')
    if failures:
        raise RuntimeError('Some input files could not be prepared:\n' + '\n'.join(failures))
    if not count:
        raise ValueError(f'No convertible images at {input_dir}/<modality>/images; archives must be extracted first')
    return count


def main():
    parser = argparse.ArgumentParser(description='Convert raster images/medical volumes into 2D restoration references')
    parser.add_argument('--input_dir', default='./datasets')
    parser.add_argument('--output_dir', default='./datasets/processed')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--modality', default='all')
    parser.add_argument('--slice_axis', type=int, default=0, help='Volume slice axis; NIfTI often uses 2')
    args = parser.parse_args()
    count = prepare(**vars(args))
    print(f'Prepared {count} slices with source-file-level train/val/test splits')


if __name__ == '__main__':
    main()
