"""Build a self-contained Colab notebook from the current project source."""
from pathlib import Path
import base64
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_archive():
    buffer = io.BytesIO()
    directories = ['data', 'models', 'training', 'inference', 'evaluation', 'utils', 'scripts', 'tests', 'configs', 'web']
    paths = [ROOT / name for name in ['README.md', 'setup.py', 'requirements.txt', 'pytest.ini', 'main.py']]
    for directory in directories:
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.suffix in {'.py', '.yaml', '.html', '.css', '.js', '.svg', '.png'})
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def build():
    payload = source_archive()
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    (ROOT / 'artifacts' / 'medical-image-restoration-source.zip').write_bytes(payload)
    cells = []
    def markdown(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.splitlines(keepends=True)})
    markdown('''# Medical image restoration — T4 training
Train LightUNet on MedMNIST chest X-rays with synthetic noise, blur and contrast corruption. Reference labels are not used. Official data splits are preserved. This is a research baseline, not evidence of clinical restoration quality.

Select **Runtime → Change runtime type → T4 GPU**, then **Run all**. The notebook includes the repaired project source and exports a portable PyTorch model for use on your local CPU. No training has been run in this notebook yet.

Source: [MedMNIST](https://medmnist.com/) / [official package](https://github.com/MedMNIST/MedMNIST). PneumoniaMNIST uses chest radiographs under the dataset's stated licensing terms.
''')
    code('''from pathlib import Path
import base64, io, os, sys, zipfile, subprocess
project = Path('/content/medical-image-restoration')
project.mkdir(parents=True, exist_ok=True)
source_bytes = base64.b64decode(''' + repr(base64.b64encode(payload).decode()) + ''')
with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
    archive.extractall(project)
(project / 'source.zip').write_bytes(source_bytes)
os.chdir(project)
sys.path.insert(0, str(project))
print('Project unpacked:', project)
''')
    code('''# Keep Colab's matching CUDA torch/torchvision installation.
requirements = [line for line in Path('requirements.txt').read_text().splitlines()
                if not line.startswith(('torch>', 'torchvision>'))]
Path('/content/restoration-requirements.txt').write_text('\\n'.join(requirements))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                '/content/restoration-requirements.txt', 'pytest'], check=True)
import torch
assert torch.cuda.is_available(), 'Enable the T4 GPU runtime, then rerun this cell.'
print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))
''')
    markdown('''## Optional: keep checkpoints in Google Drive
Set `PERSIST_TO_DRIVE = True` before running this cell to save checkpoints as training proceeds. Colab will ask you to authorize Drive. Otherwise checkpoints live in the temporary runtime until you download the final ZIP.
''')
    code('''PERSIST_TO_DRIVE = False
checkpoint_dir = project / 'checkpoints'
if PERSIST_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    checkpoint_dir = Path('/content/drive/MyDrive/medical-image-restoration/checkpoints')
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print('Checkpoints:', checkpoint_dir)
''')
    code('''# Run the source regression tests before spending GPU time on training.
subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)
from models.factory import create_model
from utils.config import load_config
# Exercise T4 mixed precision on both supported architectures.
for config_path in ['configs/colab_t4.yaml', 'configs/gpu_config.yaml']:
    model = create_model(load_config(config_path)).cuda()
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(torch.randn(1, 3, 32, 32, device='cuda'))
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    print(config_path, 'GPU forward/backward OK')
    del model, output, loss
torch.cuda.empty_cache()
''')
    code('''# Change --datasets to add ct, retinal or ultrasound for a broader experiment.
subprocess.run([sys.executable, 'scripts/download_all_datasets.py',
                '--datasets', 'xray', '--size', '128'], check=True)
''')
    markdown('''## Train and export the best model
The default is 30 epochs at 128×128 with mixed precision and batch size 16. Edit `EPOCHS` below if needed. To train the compact transformer instead, set `CONFIG = 'configs/gpu_config.yaml'` (it uses batch size 4). Training saves resumable `.ckpt` files and a portable `<model>_best.pt` with architecture/config metadata. To resume an interrupted run, set `RESUME` to the saved `last.ckpt` path.
''')
    code('''CONFIG = 'configs/colab_t4.yaml'
EPOCHS = 30
RESUME = None
command = [sys.executable, 'scripts/train.py', '--config', CONFIG,
           '--epochs', str(EPOCHS), '--output_dir', str(checkpoint_dir)]
if RESUME:
    command += ['--checkpoint', str(RESUME)]
subprocess.run(command, check=True)
from utils.config import load_config
model_name = load_config(CONFIG)['model']['name']
checkpoint_path = checkpoint_dir / f'{model_name}_best.pt'
assert checkpoint_path.is_file()
print('Portable model:', checkpoint_path)
''')
    code('''subprocess.run([sys.executable, 'scripts/evaluate.py', '--checkpoint', str(checkpoint_path),
                '--data_dir', 'datasets/processed', '--output', 'outputs/evaluation_results.json'], check=True)
import json
print(json.dumps(json.loads(Path('outputs/evaluation_results.json').read_text()), indent=2))
''')
    code('''# Prove the GPU-trained checkpoint also loads and restores images on CPU.
import numpy as np
import matplotlib.pyplot as plt
from data.io import load_image
from data.degradation import add_gaussian_noise, add_motion_blur
from utils.checkpoint import load_model
from inference.restore import RestorationPipeline
model, config = load_model(checkpoint_path, device='cpu')
reference_path = next(Path('datasets/processed/test').rglob('*.png'))
clean = load_image(reference_path)
degraded = add_motion_blur(add_gaussian_noise(clean, std=25, np_rng=np.random.default_rng(42)), kernel_size=7)
restored = RestorationPipeline(model, target_size=config['data']['target_size']).restore(degraded)['restored']
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, title, image in zip(axes, ['Reference', 'Degraded', 'Restored'], [clean, degraded, restored]):
    ax.imshow(image.squeeze(), cmap='gray' if image.shape[-1] == 1 else None)
    ax.set_title(title)
    ax.axis('off')
fig.tight_layout()
fig.savefig('outputs/comparison.png', dpi=150)
plt.show()
''')
    code('''# Download a reusable model + metrics + matching project source.
bundle = project / 'restoration_results.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.write(checkpoint_path, f'checkpoints/{checkpoint_path.name}')
    archive.write('outputs/evaluation_results.json', 'evaluation_results.json')
    archive.write('outputs/comparison.png', 'comparison.png')
    archive.write('source.zip', 'medical-image-restoration-source.zip')
from google.colab import files
files.download(str(bundle))
print('Locally: extract the model into checkpoints/, then run:')
print(f'python scripts/restore_image.py input.png --checkpoint checkpoints/{checkpoint_path.name} --output restored.png')
''')
    notebook = {'nbformat': 4, 'nbformat_minor': 0,
                'metadata': {'colab': {'provenance': [], 'gpuType': 'T4'},
                             'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                             'language_info': {'name': 'python'}}, 'cells': cells}
    destination = ROOT / 'notebooks' / 'train_colab.ipynb'
    destination.write_text(json.dumps(notebook, indent=1) + '\n')
    print(f'Built {destination}: {len(cells)} cells, {len(payload)} bytes of source')


if __name__ == '__main__':
    build()
