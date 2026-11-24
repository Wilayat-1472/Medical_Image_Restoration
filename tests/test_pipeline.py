"""Regressions for the broken config, pairing, model and inference contracts."""
import random
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from utils.config import load_config
from utils.checkpoint import load_model, save_checkpoint
from models.factory import create_model
from data.dataset import MedicalRestorationDataset, SyntheticDegradationDataset
from data.degradation import apply_mixed_degradation, add_gaussian_blur, add_salt_pepper_noise
from data.io import load_image, save_image
from data.preprocessing import MedicalTransform
from inference.restore import RestorationPipeline
from inference.failsafe import FailSafe
from evaluation.metrics import compute_all_metrics
from training.losses import CombinedLoss


def tiny_config(name='light_unet'):
    config = load_config('configs/cpu_config.yaml')
    config['model']['name'] = name
    config['model']['light_unet'].update(base_filters=8, depth=2)
    config['model']['restormer_unet'].update(dim=8, num_blocks=[1, 1, 1], num_heads=[1, 2, 4])
    config['data']['target_size'] = 32
    return config


@pytest.mark.parametrize('name', ['cpu_config', 'gpu_config', 'colab_t4'])
def test_nested_config(name):
    cfg = load_config(f'configs/{name}.yaml')
    assert cfg['model']['in_channels'] == 3
    assert cfg['training']['scheduler']['name'] == 'cosine_annealing'
    assert cfg['training']['pin_memory']


@pytest.mark.parametrize('name', ['light_unet', 'restormer_unet'])
def test_forward_backward_checkpoint(name, tmp_path):
    cfg = tiny_config(name)
    model = create_model(cfg)
    x = torch.randn(2, 3, 33, 35)
    output = model(x)
    assert output.shape == x.shape
    loss = CombinedLoss(perceptual_weight=0)(output, torch.tanh(x))['total']
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    path = tmp_path / 'model.pt'
    model.eval()
    save_checkpoint(path, model, cfg)
    restored, saved_config = load_model(path)
    torch.testing.assert_close(model(x), restored(x))
    assert saved_config == cfg


def test_synthetic_corruption_range_and_channels():
    image = np.full((40, 40, 1), 128, dtype=np.uint8)
    for seed in range(40):
        result = apply_mixed_degradation(image, rng=random.Random(seed), np_rng=np.random.default_rng(seed))
        assert result['degraded'].shape == image.shape
        assert 0 <= result['degraded'].min() <= result['degraded'].max() <= 1
    assert add_gaussian_blur(image.astype(np.float32) / 255).shape == image.shape
    assert add_salt_pepper_noise(image).shape == image.shape


def test_split_alignment_and_repeatable_validation(tmp_path):
    for i in range(30):
        image = np.random.default_rng(i).random((40, 40, 3), dtype=np.float32)
        for folder in ['clean', 'degraded']:
            save_image(tmp_path / 'xray' / folder / f'{i}.png', image)
    sets = [MedicalRestorationDataset(tmp_path, split=s, transform=MedicalTransform(target_size=(32, 32), augment=s == 'train'))
            for s in ['train', 'val', 'test']]
    paths = [{p[0] for p in ds.pairs} for ds in sets]
    assert sum(map(len, paths)) == 30
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    for _ in range(5):
        batch = sets[0][0]
        torch.testing.assert_close(batch['clean'], batch['degraded'])
    torch.testing.assert_close(sets[1][0]['degraded'], sets[1][0]['degraded'])
    (tmp_path / 'xray' / 'degraded' / '0.png').unlink()
    other = MedicalRestorationDataset(tmp_path, split='train')
    assert len(other) == 24


def test_official_splits_and_synthetic_validation(tmp_path):
    for split in ['train', 'val', 'test']:
        for i in range(3):
            image = np.random.default_rng(i).random((40, 40, 3), dtype=np.float32)
            save_image(tmp_path / split / 'xray' / 'clean' / f'{i}.png', image)
    ds = MedicalRestorationDataset(tmp_path, split='val')
    assert len(ds) == 3
    torch.testing.assert_close(ds[0]['degraded'], ds[0]['degraded'])
    assert all('/val/' in p for p, _ in ds.pairs)


def test_explicit_image_list_outside_cwd_and_relocated_validation(tmp_path):
    paths = []
    for index in range(30):
        # Include train/val/test directory names without opting into directory
        # split discovery: this API's explicit image list remains authoritative.
        path = tmp_path / 'original' / ('train', 'val', 'test')[index % 3] / f'{index}.png'
        save_image(path, np.random.default_rng(index).random((32, 32, 3), dtype=np.float32))
        paths.append(path)
    ds = SyntheticDegradationDataset(iter(paths), split='val')
    assert len(ds) == 3
    expected = ds[0]['degraded']
    torch.testing.assert_close(expected, ds[0]['degraded'])
    import shutil
    shutil.copytree(tmp_path / 'original', tmp_path / 'relocated')
    moved = [tmp_path / 'relocated' / p.relative_to(tmp_path / 'original') for p in paths]
    other = SyntheticDegradationDataset(moved, split='val')
    torch.testing.assert_close(expected, other[0]['degraded'])
    with pytest.raises(ValueError, match='at least one'):
        SyntheticDegradationDataset([])


def test_inference_shapes_batch_and_validation():
    pipeline = RestorationPipeline(create_model(tiny_config()), target_size=32)
    images = [np.random.default_rng(i).random((35+i, 41+i, 3), dtype=np.float32) for i in range(2)]
    for image, result in zip(images, pipeline.restore_batch(images)):
        assert result['restored'].shape == image.shape
        assert 0 <= result['restored'].min() <= result['restored'].max() <= 1
    output = pipeline.restore(torch.from_numpy(images[0].transpose(2, 0, 1)), return_numpy=False)
    assert output['restored_tensor'].shape == (3, 35, 41)
    with pytest.raises(ValueError):
        pipeline.restore(np.ones((40, 40), dtype=np.float32))
    assert not FailSafe().validate_input(np.empty((0, 0)))['valid']


def test_uint16_and_metric_range(tmp_path):
    values = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16)
    path = tmp_path / '16bit.png'
    cv2.imwrite(str(path), values)
    result = load_image(path)
    assert result.max() == 1 and result.min() == 0
    pred, target = torch.full((3, 16, 16), -0.5), torch.full((3, 16, 16), -1.)
    metrics = compute_all_metrics(pred, target)
    assert metrics['mse'] == pytest.approx(0.25**2)
    assert metrics['psnr'] == pytest.approx(12.0411998)


def test_strict_checkpoint_rejects_missing_weights(tmp_path):
    cfg = tiny_config()
    torch.save({'state_dict': {}, 'config': cfg}, tmp_path / 'bad.pt')
    with pytest.raises(RuntimeError):
        load_model(tmp_path / 'bad.pt')


def test_medical_volume_conversion(tmp_path):
    from scripts.prepare_data import read_slices, prepare
    import nibabel as nib
    import h5py
    volume = np.arange(3 * 36 * 40, dtype=np.float32).reshape(3, 36, 40)
    source = tmp_path / 'raw' / 'mri' / 'images'
    source.mkdir(parents=True)
    np.save(source / 'volume.npy', volume)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), source / 'volume.nii.gz')
    with h5py.File(source / 'volume.h5', 'w') as handle:
        handle.create_dataset('reconstruction_rss', data=volume)
    for filename in ['volume.npy', 'volume.nii.gz', 'volume.h5']:
        slices = list(read_slices(source / filename))
        assert len(slices) == 3
        assert all(x.shape == (36, 40) and x.min() >= 0 and x.max() <= 1 for x in slices)
    assert prepare(tmp_path / 'raw', tmp_path / 'processed', target_size=32, modality='mri') == 9
    assert len(list((tmp_path / 'processed').rglob('*.png'))) == 9
    with pytest.raises(ValueError, match='No convertible images'):
        prepare(tmp_path / 'missing', tmp_path / 'empty')


def test_explanation_and_reports(tmp_path):
    from evaluation.explainability import visualize_explanations
    from inference.report import generate_report, save_report
    from inference.confidence import ConfidenceScorer
    cfg = tiny_config()
    model = create_model(cfg).eval()
    image = np.random.default_rng(1).random((32, 32, 3), dtype=np.float32)
    x = MedicalTransform(target_size=(32, 32))(image).unsqueeze(0)
    result = visualize_explanations(model, x, save_path=str(tmp_path / 'explanation.png'))
    assert result['attention_map'].shape == (32, 32)
    assert (tmp_path / 'explanation.png').is_file()
    score = ConfidenceScorer().score(image, image)
    assert score['calibrated'] is False
    report = generate_report('input.png', {}, score)
    save_report(report, str(tmp_path / 'reports' / 'report.html'), format='html', degraded_image=image, restored_image=image)
    assert (tmp_path / 'reports' / 'report.html').is_file()


def test_dicom_and_kspace_conversion(tmp_path):
    from scripts.prepare_data import read_slices
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    import h5py
    metadata = FileMetaDataset()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    metadata.MediaStorageSOPClassUID = generate_uid()
    metadata.MediaStorageSOPInstanceUID = generate_uid()
    path = tmp_path / 'image.dcm'
    ds = FileDataset(str(path), {}, file_meta=metadata, preamble=b'\0' * 128)
    ds.Rows, ds.Columns = 36, 40
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    pixels = np.arange(36 * 40, dtype=np.uint16).reshape(36, 40)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    image = next(read_slices(path))
    assert image.shape == pixels.shape and image.max() == 1
    kspace = np.zeros((2, 3, 36, 40), dtype=np.complex64)
    kspace[..., 18, 20] = 1
    with h5py.File(tmp_path / 'coils.h5', 'w') as handle:
        handle.create_dataset('kspace', data=kspace)
    reconstructed = list(read_slices(tmp_path / 'coils.h5'))
    assert len(reconstructed) == 2
    assert all(x.shape == (36, 40) and np.isfinite(x).all() for x in reconstructed)


def test_classical_benchmark_and_stress_tests():
    from evaluation.benchmark import benchmark_inference_time, compare_with_classical
    from evaluation.edge_cases import EdgeCaseTester
    cfg = tiny_config()
    model = create_model(cfg).eval()
    clean = np.random.default_rng(3).random((32, 32, 1), dtype=np.float32)
    comparison = compare_with_classical(model, [clean * 0.8], [clean])
    assert set(comparison) == {'median_filter', 'gaussian_filter', 'wiener_filter', 'bilateral_filter', 'deep_learning_model'}
    assert all(np.isfinite(scores['psnr']) and np.isfinite(scores['ssim']) for scores in comparison.values())
    speed = benchmark_inference_time(model, (1, 3, 32, 32), num_runs=2, num_warmup=0)
    assert speed['mean_ms'] > 0
    tester = EdgeCaseTester(model, target_size=32)
    result = tester.test_noise_levels([clean], noise_levels=[10])
    assert np.isfinite(result['gaussian_noise_std_10']['psnr'])
    with pytest.raises(ValueError):
        tester.run_all([])
