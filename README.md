# Low-Dose CT Residual Attention U-Net Restoration

Paired Low-Dose CT (LDCT) to Full-Dose CT (FDCT) restoration using a **Residual Attention U-Net**, trained on GPU (Google Colab T4) and exported as a compressed `.joblib` bundle for local CPU inference.

## Repository Structure

```text
low-dose-ct-residual-restoration/
├── assets/
│   └── ldct_clinical_diagnostics.png
├── data/
│   └── ldct_test_metrics.csv
├── models/
│   └── ldct_restorer_bundle.joblib      # ~7.5 MB compressed CPU bundle
├── notebooks/
│   └── ldct_residual_attention_unet.ipynb
├── src/
│   ├── __init__.py
│   └── model.py                         # ConvBlock, AttentionGate, ResidualAttentionUNet
├── predict.py                           # Local CPU inference script for any LDCT slice
├── requirements.txt
└── README.md
```

## Local CPU Inference

Install dependencies:
```bash
pip install -r requirements.txt
```

Restore any LDCT slice (`.png`, `.jpg`, `.tif`, `.dcm`, or `.npy`):
```bash
python predict.py \
  --input path/to/low_dose_slice.png \
  --bundle models/ldct_restorer_bundle.joblib \
  --output restored.png \
  --compare comparison.png
```

Or launch the local web interface:
```bash
python app.py --bundle models/ldct_restorer_bundle.joblib
```
Open **http://127.0.0.1:7860** in your browser.
