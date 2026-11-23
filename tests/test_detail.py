"""Quality regressions: do not erase native-resolution detail during restoration."""
import numpy as np
import pytest
import torch
from inference.restore import RestorationPipeline
from evaluation.detail import compare_detail
from training.losses import CombinedLoss


class MustNotRun(torch.nn.Module):
    def forward(self, x):
        raise AssertionError('Zero strength must not run the model')


@pytest.mark.parametrize('channels', [1, 3])
def test_zero_strength_keeps_original_exactly(channels):
    image = np.random.default_rng(3).random((81, 119, channels), dtype=np.float32)
    pipeline = RestorationPipeline(MustNotRun(), target_size=32)
    result = pipeline.restore(image, strength=0)
    np.testing.assert_array_equal(result['restored'], image)


def test_identity_model_does_not_erase_high_resolution_detail():
    y, x = np.mgrid[:171, :283]
    image = (0.25 + 0.5 * ((x + y) % 2)).astype(np.float32)[..., None]
    pipeline = RestorationPipeline(torch.nn.Identity(), target_size=32)
    result = pipeline.restore(image, strength=1)['restored']
    assert result.shape == image.shape
    np.testing.assert_array_equal(result, image)


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_subpixel_details_survive_low_resolution_model():
    y, x = np.mgrid[:256, :512]
    image = (0.25 + 0.5 * ((x + y) % 2)).astype(np.float32)[..., None]
    pipeline = RestorationPipeline(ConstantModel(), target_size=32)
    result = pipeline.restore(image, strength=1)['restored']
    # The proxy and prediction are uniform; replacement used to erase the grid.
    # The correction path must retain its original full-resolution contrast.
    assert result.std() == pytest.approx(image.std(), abs=1e-6)
    np.testing.assert_allclose(result, image, atol=1e-6)


def test_strength_interpolates_at_model_resolution():
    image = np.random.default_rng(5).uniform(0.15, 0.85, (32, 32, 3)).astype(np.float32)
    pipeline = RestorationPipeline(ConstantModel(), target_size=32)
    for strength in [0.25, 0.5, 1.0]:
        result = pipeline.restore(image, strength=strength)['restored']
        np.testing.assert_allclose(result, image + strength * (0.5 - image), atol=1e-7)


@pytest.mark.parametrize('strength', [-1, 1.1, float('nan'), float('inf')])
def test_invalid_strength(strength):
    with pytest.raises(ValueError, match='strength'):
        RestorationPipeline(ConstantModel(), strength=strength)


def test_detail_check_warns_about_lost_edges():
    image = np.zeros((64, 64, 1), dtype=np.float32)
    image[:, 32:] = 1
    blurred = np.ones_like(image) * 0.5
    assert compare_detail(image, blurred)['weaker_edges']
    assert not compare_detail(image, image)['weaker_edges']
    assert compare_detail(blurred, blurred)['edge_energy_ratio'] is None


def test_edge_loss_penalizes_smoothing_and_has_finite_gradient():
    target = torch.full((1, 3, 32, 32), -1.)
    target[..., 16:] = 1
    blurred = torch.nn.functional.avg_pool2d(target, 7, 1, 3).requires_grad_()
    loss = CombinedLoss(l1_weight=0, ssim_weight=0, perceptual_weight=0,
                        charbonnier_weight=0, edge_weight=1)
    assert loss(target, target)['edge'] == 0
    result = loss(blurred, target)
    assert result['edge'] > 0
    result['total'].backward()
    assert torch.isfinite(blurred.grad).all()


def test_identity_examples_are_train_only(tmp_path):
    from data.dataset import MedicalRestorationDataset
    from data.io import save_image
    from data.preprocessing import MedicalTransform
    for split in ['train', 'val', 'test']:
        save_image(tmp_path / split / 'clean' / 'image.png',
                   np.random.default_rng(3).random((40, 40, 1), dtype=np.float32))
    transform = MedicalTransform(target_size=(32, 32), augment=True)
    train = MedicalRestorationDataset(tmp_path, split='train', transform=transform, identity_probability=1)
    example = train[0]
    torch.testing.assert_close(example['clean'], example['degraded'])
    assert example['degradations'] == 'identity'
    validation = MedicalRestorationDataset(tmp_path, split='val', identity_probability=1)
    assert validation[0]['degradations'] != 'identity'
