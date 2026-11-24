# Medical Image Restoration

A PyTorch research project that learns to reconstruct reference medical images from synthetically degraded copies. It learns corrections for synthetic noise, blur, low contrast and JPEG artifacts; improvement on a particular upload is not guaranteed. It does not diagnose disease or segment anatomy. No pretrained weights are bundled with the source.

The default experiment uses **MedMNIST PneumoniaMNIST chest X-rays**, retaining the provider's train/validation/test splits. These are reference images for synthetic restoration, not paired real-world low-dose/high-dose acquisitions. Results on this task do not demonstrate performance on other modalities or clinical images.

## Current trained model

This workspace contains a **LightUNet trained for 30 epochs on a Colab Tesla T4**, using 128×128 PneumoniaMNIST chest X-rays. Its portable checkpoint is:

```text
checkpoints/colab_t4/light_unet_best.pt
```

The recorded evaluation on 624 held-out, synthetically degraded images reported **28.74 dB PSNR** and **0.778 SSIM**, compared with **22.71 dB** and **0.504** for the degraded inputs. These are **raw-model results at 128px**, not measurements of the current detail-preserving browser output or full-resolution medical restoration. The blur fix changes inference; the existing trained weights are unchanged.

Checkpoints, datasets and generated reports are ignored by Git and are not bundled in source-only copies. A fresh checkout needs a trained checkpoint before launching the interface. The separate `configs/detail_finetune.yaml` recipe has not been trained.

## Install

Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
# CPU-only PyTorch; omit this command on Colab, which supplies CUDA PyTorch.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
pip install pytest
pytest -q
```

OpenCV is headless by default. The interactive demo uses terminal prompts and saves comparison PNGs. To use OpenCV GUI windows, install the GUI build instead of the headless build.

## Browser interface

Launch the local interface with the T4-trained model:

```bash
.venv/bin/python scripts/web_app.py
```

Open **http://127.0.0.1:7860** in your browser. Upload or drop a PNG, JPG, TIFF or BMP image, or try the sample X-ray. Start with **25–50% restoration strength**, select **Restore image**, use the before/after slider, and download the restored PNG. If you change strength after restoring, select **Update preview** to apply the new setting; the displayed result keeps its previous setting until you do so. The app loads the trained model once; uploads stay in memory and are not saved to disk.

The default checkpoint is `checkpoints/colab_t4/light_unet_best.pt`. To use another checkpoint or port:

```bash
.venv/bin/python scripts/web_app.py --checkpoint checkpoints/light_unet_best.pt --port 7861
```

The server listens on this computer only. It accepts single images from 32×32 through 4096×4096 pixels, up to 20 MB, and scales 16-bit grayscale inputs using their full intensity range. Downloads are **8-bit PNG previews** and do not retain the original image metadata. Medical volumes must be converted first. The interface includes loading states, validation errors, a comparison slider, keyboard controls, and a guide describing the model's 128px correction and chest X-ray training scope. It uses the existing Python dependencies; no frontend build or extra UI framework is needed.

## Train and use locally

```bash
# Downloads reference images directly into datasets/processed/{train,val,test}/xray/clean.
python scripts/download_all_datasets.py --datasets xray --size 128
python scripts/train.py --config configs/cpu_config.yaml

# Best validation weights, architecture and preprocessing are saved together.
python scripts/restore_image.py input.png --checkpoint checkpoints/light_unet_best.pt --output restored.png --report report.html
python scripts/batch_restore.py input_folder --checkpoint checkpoints/light_unet_best.pt
python scripts/evaluate.py --checkpoint checkpoints/light_unet_best.pt --data_dir datasets/processed
python scripts/benchmark.py --checkpoint checkpoints/light_unet_best.pt
```

`python main.py` downloads the default X-ray references and trains with the CPU/GPU config chosen from CUDA availability. Use `--skip-download --skip-prepare` for already prepared data, and `--config` to choose an architecture explicitly. Run commands from the repository root.

Standard evaluation reports raw-model PSNR, SSIM, MSE, MAE, and PSNR/SSIM improvement over synthetic degraded inputs at the training resolution. These metrics do not measure full-resolution detail preservation or the blended browser output. Inputs and predictions are correctly converted from the network's [-1, 1] range before metrics. Benchmark timing synchronizes CUDA. Recorded results apply only to their stated experiment and resolution.

## Detail preservation and blur

The original inference path resized every upload to 128×128 and enlarged the prediction. This can visibly blur larger inputs even if low-resolution benchmark metrics improve. The current path applies the model's **correction** to the original image instead:

```
output = original + strength × upsample(prediction − resized_original)
```

The default strength is **0.5** (50%). Start at **25%** for an already fairly clean image, then compare the result with the original before increasing it.

| Strength | Behavior |
| --- | --- |
| 0% | Apply no model correction; PNG export still uses 8-bit output. |
| 25% | Apply a gentle correction with more of the original retained. |
| 50% | Default correction strength. |
| 100% | Apply the full learned correction; this can still over-process an image. |

The web interface has a strength slider; select **Update preview** after changing it. At 0% the original pixels are returned without running the model. Grayscale inputs stay grayscale. The CLI exposes the same setting:

```bash
.venv/bin/python scripts/restore_image.py input.png --checkpoint checkpoints/colab_t4/light_unet_best.pt --strength 0.25 --output restored.png
```

Batch restoration accepts the same option:

```bash
.venv/bin/python scripts/batch_restore.py input_folder --checkpoint checkpoints/colab_t4/light_unet_best.pt --strength 0.25 --output_dir outputs/restored
```

This avoids replacing fine-resolution detail with an enlarged prediction; it does **not** make the 128px model a full-resolution denoiser. Fine noise may remain. At the model resolution, an aggressive correction can still soften edges. Lower strength is appropriate when the original is already fairly clean. The edge check warns about substantial attenuation but cannot distinguish all noise removal from detail loss.

Run the separate quality audit to inspect clean-input changes, mild versus strong corruption, and a controlled resolution pattern:

```bash
.venv/bin/python scripts/audit_restoration.py
```

Reports and comparison figures are saved in `outputs/detail_audit/`. The default audit uses 24 existing held-out reference PNGs, records their actual dimensions, and compares the previous replacement path with correction strengths of 25%, 50% and 100%. Its small-reference results do not establish quality on unseen high-resolution medical images.

The recorded audit used **24 references at 64×64**, a 128px model proxy, and a separate non-medical 512×384 test pattern:

| Check | Previous replacement | Current correction at 50% |
| --- | ---: | ---: |
| Clean-reference SSIM | 0.9876 | 0.9955 |
| Mild Gaussian blur, reference PSNR | 36.73 dB | 40.38 dB |
| Gaussian noise, σ=25, reference PSNR | 27.28 dB | 22.25 dB |
| Controlled high-resolution pattern SSIM | 0.0295 | 0.9470 |

The noise row shows the tradeoff: retaining fine detail can retain fine noise and reduce denoising performance. This is not an across-the-board model improvement. The pattern demonstrates the resizing defect; it does not establish high-resolution medical accuracy. See the generated `outputs/detail_audit/quality_audit.json` and `outputs/detail_audit/detail_comparison.png` when available locally.

### Fine-tuning for detail preservation

`configs/detail_finetune.yaml` is an **untrained fine-tuning recipe** for the existing architecture: milder synthetic corruption, 20% clean/identity training examples, and a target-gradient loss. It does not change the supplied checkpoint. On a GPU runtime, initialize the existing weights and write the new experiment separately:

```bash
python scripts/train.py --config configs/detail_finetune.yaml --init_weights checkpoints/colab_t4/light_unet_best.pt --output_dir checkpoints/detail_finetune
```

`--init_weights` starts a new optimizer for fine-tuning; `--checkpoint` resumes an existing Lightning training run. Validation/test data always retain synthetic corruptions; identity examples are train-only. The original severe-corruption recipe remains available. Better full-resolution restoration will require higher-resolution training references and separate detail-preservation validation, not simply increasing an inference image-size setting.

## Train on Colab T4

Open `notebooks/train_colab.ipynb` in Colab. It contains a snapshot of the project, so no GitHub repository or source upload is required. Select **Runtime → Change runtime type → T4 GPU**, then run all cells. The notebook checks CUDA availability before training.

The default `configs/colab_t4.yaml` trains LightUNet on 128px X-ray references with mixed precision, batch size 16 and 30 epochs. It exports the best validation checkpoint, evaluates the held-out test split, saves an example comparison, and downloads a ZIP containing the `.pt` model, metrics and project source. For the default browser-interface path, extract the model to `checkpoints/colab_t4/light_unet_best.pt`. If you keep it at `checkpoints/light_unet_best.pt`, pass that path with `--checkpoint`. A runtime disconnection can lose files; the notebook includes an optional Drive-persistence cell before training.

The T4 checkpoint described above was produced by a completed run. The local notebook file is a generated source snapshot, so changing the project does not automatically update its embedded code or a copy already open in Colab. Regenerate it before a new run, then open the regenerated notebook in Colab:

```bash
python scripts/build_colab_notebook.py
```

`configs/gpu_config.yaml` instead selects a compact Restormer-style U-Net (dim 24, two blocks per level, batch size 4). Both models support forward/backward passes and arbitrary image dimensions through the inference interface. Their weights are not interchangeable.

## Data

Automatic downloads use [MedMNIST](https://medmnist.com/) through its [official Python package](https://github.com/MedMNIST/MedMNIST):

| CLI modality | Dataset | Reference type |
|---|---|---|
| xray | PneumoniaMNIST | Chest radiographs |
| ct | OrganAMNIST | Abdominal CT slices |
| retinal | RetinaMNIST | Fundus photographs |
| ultrasound | BreastMNIST | Breast ultrasound |

For example: `python scripts/download_all_datasets.py --datasets xray retinal --size 128`. The downloader preserves official splits and ignores classification labels. MedMNIST's licensing and source attribution apply. There is no automatic MRI download; obtain authorized fastMRI data separately from https://fastmri.med.nyu.edu/.

For your own data, prefer explicit patient-separated splits:

```
datasets/processed/
  train/<modality>/clean/*.png
  val/<modality>/clean/*.png
  test/<modality>/clean/*.png
```

Each `clean` directory can have a sibling `degraded` directory containing matching relative filenames. Missing counterparts get synthetic degradation. A flat `clean/` tree or modality/clean trees are also accepted; they use deterministic 80/10/10 **image-level** splits. Use explicit patient-separated splits for studies with multiple images per patient.

`SyntheticDegradationDataset(image_paths, split="val")` also accepts absolute paths
or an iterable of paths outside the working directory. The supplied list is
authoritative, even if its common parent contains directories named `train`,
`val`, or `test`. Validation/test corruption seeds use paths relative to that
common parent, so moving the same image tree preserves the synthetic evaluation.

For local source data, put files in `datasets/<modality>/images/`, then run:

```bash
python scripts/prepare_data.py --modality mri --target_size 128 --slice_axis 0
```

Conversion supports ordinary rasters, `.dcm`, `.nii[.gz]`, `.npy` volumes and fastMRI `.h5` reconstructions/k-space. Extract archives first. NIfTI often needs `--slice_axis 2`. Volumes are split by source file so slices from one volume stay together; this does not infer patient IDs. Choose explicit patient splits when multiple volumes belong to a patient. Medical-volume intensities use percentile scaling and are exported as 8-bit 2D PNGs; this loses quantitative intensity units and volume metadata. Restoration accepts 2D raster files or normalized arrays, not raw volumes.

## Verification

The latest code check passed **45 automated tests**, covering model forward/backward passes, checkpoint loading, training and fine-tuning, image conversion, aligned pairs, upload validation, HTTP endpoints, and detail preservation. The quality regressions include exact original-pixel preservation at 0%, high-resolution detail surviving the proxy path, valid strength ranges, and train-only identity examples.

Run the automated suite from the project root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

The live Chrome checks passed upload, drag-and-drop, actual restoration, strength updates, comparison modes, keyboard slider use, PNG downloads, error recovery and the mobile layout. To repeat them, start the interface in one terminal and, with Google Chrome installed, run in another:

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python tests/browser_smoke.py
```

Screenshots are written to `outputs/ui/`. Passing functional tests does not establish restoration quality on a particular upload; use the quality audit and compare fine details with the original.

## Implementation

- `data/`: conversion, shared image I/O, aligned augmentation, synthetic corruption and split-aware loaders.
- `models/`: depthwise separable LightUNet and Restormer-style encoder/decoder.
- `training/`: Lightning training with L1, SSIM, optional ImageNet VGG19 perceptual, Charbonnier and target-gradient losses; optional initialization from existing weights for fine-tuning.
- `inference/`: validation before inference, detail-preserving corrections, adjustable strength, grayscale preservation, batch restoration, heuristic checks and JSON/HTML reports.
- `evaluation/`: reference metrics, edge-energy checks, classical-filter comparisons, corruption stress tests and gradient-based visualizations. `scripts/audit_restoration.py` compares clean inputs, corruption types and detail retention.
- `utils/`: recursive config inheritance and strict, portable checkpoint loading.
- `web/`: local browser interface and upload/restoration API backed by the trained model.

CPU and Colab baseline configs disable VGG to avoid an external weight download. The full GPU config enables it. VGG inputs use ImageNet normalization. The separate quality-assessment and uncertainty-head modules are experimental and are not trained or integrated. Confidence is a **non-calibrated image-consistency heuristic**, not a probability of anatomical correctness. A Grad-CAM overlay is model sensitivity, not a verified anatomical explanation.

Portable `.pt` files contain `state_dict`, config and validation metadata and load on CPU or GPU using `weights_only=True`. Lightning `.ckpt` files additionally support resuming training with `--checkpoint`. Legacy bare state dicts need the matching `--config`. Loading is strict; incompatible weights raise an error. PyTorch serialization is used instead of joblib because the artifact is a PyTorch neural network.
