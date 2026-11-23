"""Exercise the real Lightning training, checkpoint export and CLI integration."""
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml
from data.io import save_image
from utils.config import load_config
from utils.checkpoint import load_model


def test_training_export_and_restore_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / 'data'
    for split in ['train', 'val', 'test']:
        for i in range(2):
            save_image(data / split / 'clean' / f'{i}.png', np.random.default_rng(i).random((32, 32, 3), dtype=np.float32))
    cfg = load_config(root / 'configs/cpu_config.yaml')
    cfg['model']['light_unet'].update(base_filters=8, depth=2)
    cfg['training'].update(max_epochs=1, batch_size=2, num_workers=0, num_threads=1, log_every_n_steps=1)
    cfg['data']['target_size'] = 32
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(tmp_path / 'mpl'))
    def run(script, *args):
        result = subprocess.run([sys.executable, str(root / 'scripts' / script), *map(str, args)],
                                cwd=root, env=env, text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    run('train.py', '--config', config, '--data_dir', data, '--output_dir', tmp_path / 'checkpoints')
    checkpoint = tmp_path / 'checkpoints/light_unet_best.pt'
    model, loaded_config = load_model(checkpoint)
    assert loaded_config['model']['light_unet']['base_filters'] == 8
    assert torch.isfinite(model(torch.randn(1, 3, 32, 32))).all()
    run('restore_image.py', data / 'test/clean/0.png', '--checkpoint', checkpoint,
        '--output', tmp_path / 'restored.png', '--report', tmp_path / 'report.html', '--explain')
    assert (tmp_path / 'restored.png').is_file()
    assert (tmp_path / 'restored_explanation.png').is_file()
    assert '<!DOCTYPE html>' in (tmp_path / 'report.html').read_text()
    run('evaluate.py', '--checkpoint', checkpoint, '--data_dir', data, '--output', tmp_path / 'metrics.json')
    assert (tmp_path / 'metrics.json').is_file()

    cfg['loss']['edge_weight'] = 0.15
    cfg['data']['identity_probability'] = 0.2
    config.write_text(yaml.safe_dump(cfg))
    run('train.py', '--config', config, '--data_dir', data, '--output_dir', tmp_path / 'finetuned',
        '--init_weights', checkpoint)
    _, fine_config = load_model(tmp_path / 'finetuned/light_unet_best.pt')
    assert fine_config['loss']['edge_weight'] == 0.15
