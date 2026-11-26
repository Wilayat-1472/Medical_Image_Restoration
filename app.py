"""Local CPU web interface for Low-Dose CT Residual Attention U-Net restoration."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import cv2
import joblib
import numpy as np

from predict import load_bundle, load_ldct_slice, restore_slice

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LDCT Residual Attention U-Net</title>
<style>
  :root {
    --bg: #0b1120; --card: #111827; --border: #1f2937;
    --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8; --accent-hover: #0ea5e9;
  }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem 1rem; }
  .wrap { max-width: 1000px; margin: 0 auto; }
  header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }
  h1 { font-size: 1.5rem; margin: 0; }
  .badge { font-size: 0.85rem; color: var(--accent); background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.25); padding: 0.3rem 0.75rem; border-radius: 999px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.25rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; }
  .drop { border: 2px dashed #334155; border-radius: 10px; padding: 1.25rem; text-align: center; cursor: pointer; flex: 1 1 260px; transition: border-color .15s; }
  .drop.drag { border-color: var(--accent); background: rgba(56,189,248,0.06); }
  .drop input { display: none; }
  .slider-group { display: flex; flex-direction: column; gap: 0.35rem; min-width: 190px; }
  .slider-group label { font-size: 0.9rem; color: var(--muted); }
  button { background: var(--accent); color: #0b1120; border: none; padding: 0.65rem 1.25rem; border-radius: 8px; font-weight: 600; cursor: pointer; }
  button:hover:not(:disabled) { background: var(--accent-hover); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  a.btn-dl { display: none; padding: 0.6rem 1.1rem; border-radius: 8px; border: 1px solid var(--accent); color: var(--accent); text-decoration: none; font-weight: 600; font-size: 0.9rem; }
  .status { font-size: 0.9rem; color: var(--muted); margin-top: 0.75rem; min-height: 1.2em; }
  .status.err { color: #f87171; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
  .panel h3 { margin: 0 0 0.75rem; font-size: 0.95rem; color: var(--muted); font-weight: 600; }
  .panel img { width: 100%; border-radius: 8px; background: #000; display: block; min-height: 200px; object-fit: contain; }
  .compare-wrap { position: relative; width: 100%; max-width: 512px; margin: 0 auto; overflow: hidden; border-radius: 8px; background: #000; display: none; }
  .compare-wrap img { width: 100%; display: block; user-select: none; }
  .compare-overlay { position: absolute; inset: 0; overflow: hidden; width: 50%; border-right: 2px solid var(--accent); }
  .compare-overlay img { width: 100%; max-width: none; }
  .wipe-bar { width: 100%; max-width: 512px; margin: 0.75rem auto 0; display: none; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Low-Dose CT Restoration</h1>
    <span class="badge" id="metrics-badge">__METRICS_BADGE__</span>
  </header>

  <div class="card">
    <div class="controls">
      <label class="drop" id="drop">
        <strong id="file-label">Drop LDCT slice or click to browse</strong>
        <div style="font-size:0.8rem;color:var(--muted);margin-top:0.25rem;">PNG, JPG, TIFF, BMP, DICOM (.dcm), or NumPy (.npy)</div>
        <input type="file" id="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.dcm,.npy">
      </label>
      <div class="slider-group">
        <label>Restoration Strength: <strong id="sval" style="color:var(--text)">1.00</strong></label>
        <input type="range" id="strength" min="0" max="1" step="0.05" value="1.0">
      </div>
      <button id="run" disabled>Restore Slice</button>
      <a id="download" class="btn-dl" download="restored_ldct.png">Download PNG</a>
    </div>
    <div class="status" id="status">Select a low-dose CT image to begin.</div>
  </div>

  <div class="card" id="wipe-card" style="display:none">
    <h3 style="margin:0 0 0.75rem;font-size:0.95rem;color:var(--muted);">Interactive Wipe (Left: Low-Dose Input | Right: Restored)</h3>
    <div class="compare-wrap" id="compare-wrap">
      <img id="cmp-after" alt="Restored CT">
      <div class="compare-overlay" id="cmp-overlay"><img id="cmp-before" alt="Low-Dose CT"></div>
    </div>
    <input type="range" id="wipe" class="wipe-bar" min="0" max="100" value="50">
  </div>

  <div class="grid">
    <div class="card panel"><h3>Low-Dose Input</h3><img id="inp" alt=""></div>
    <div class="card panel"><h3>Restored CT</h3><img id="out" alt=""></div>
  </div>
</div>

<script>
const fileEl = document.getElementById('file'), dropEl = document.getElementById('drop'),
      fileLabel = document.getElementById('file-label'), strengthEl = document.getElementById('strength'),
      svalEl = document.getElementById('sval'), runBtn = document.getElementById('run'),
      dlLink = document.getElementById('download'), statusEl = document.getElementById('status'),
      inpImg = document.getElementById('inp'), outImg = document.getElementById('out'),
      wipeCard = document.getElementById('wipe-card'), cmpWrap = document.getElementById('compare-wrap'),
      cmpOverlay = document.getElementById('cmp-overlay'), cmpBefore = document.getElementById('cmp-before'),
      cmpAfter = document.getElementById('cmp-after'), wipeBar = document.getElementById('wipe');

let selectedFile = null;

function setStatus(msg, isErr=false) {
  statusEl.textContent = msg;
  statusEl.className = isErr ? 'status err' : 'status';
}

function selectFile(f) {
  if (!f) return;
  selectedFile = f;
  fileLabel.textContent = f.name;
  runBtn.disabled = false;
  setStatus(`Ready: ${f.name} (${(f.size / 1024).toFixed(1)} KB)`);
}

fileEl.onchange = () => selectFile(fileEl.files[0]);
dropEl.ondragover = (e) => { e.preventDefault(); dropEl.classList.add('drag'); };
dropEl.ondragleave = () => dropEl.classList.remove('drag');
dropEl.ondrop = (e) => { e.preventDefault(); dropEl.classList.remove('drag'); selectFile(e.dataTransfer.files[0]); };
strengthEl.oninput = () => { svalEl.textContent = Number(strengthEl.value).toFixed(2); };
wipeBar.oninput = () => { cmpOverlay.style.width = wipeBar.value + '%'; };

function syncOverlayWidth() {
  if (cmpWrap.clientWidth > 0) cmpBefore.style.width = cmpWrap.clientWidth + 'px';
}
window.addEventListener('resize', syncOverlayWidth);

runBtn.onclick = async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  setStatus('Restoring CT slice on CPU...');
  try {
    const buf = await selectedFile.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
      bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    const res = await fetch('/api/restore', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        filename: selectedFile.name,
        data: btoa(bin),
        strength: Number(strengthEl.value)
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Restoration failed');
    inpImg.src = data.input;
    outImg.src = data.restored;
    cmpBefore.src = data.input;
    cmpAfter.src = data.restored;
    wipeCard.style.display = 'block';
    cmpWrap.style.display = 'block';
    wipeBar.style.display = 'block';
    cmpAfter.onload = syncOverlayWidth;
    dlLink.href = data.restored;
    dlLink.style.display = 'inline-block';
    setStatus(`Restored ${selectedFile.name} (${data.width}x${data.height}) at strength ${Number(strengthEl.value).toFixed(2)}.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    runBtn.disabled = false;
  }
};
</script>
</body>
</html>"""


def _encode_png_data_url(image: np.ndarray) -> str:
    u8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def create_handler(model, config: dict, metrics_badge: str):
    html_bytes = HTML_TEMPLATE.replace("__METRICS_BADGE__", metrics_badge).encode("utf-8")
    target_size = config.get("image_size", 256)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b'{"status":"ok"}', "application/json")
            else:
                self._send(200, html_bytes, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/api/restore":
                self._send(404, b'{"error":"Not found"}', "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                filename = Path(payload.get("filename", "slice.png")).name
                suffix = Path(filename).suffix.lower() or ".png"
                raw_bytes = base64.b64decode(payload["data"])
                strength = float(payload.get("strength", 1.0))

                with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                    tmp.write(raw_bytes)
                    tmp.flush()
                    ldct = load_ldct_slice(tmp.name)

                restored = restore_slice(model, ldct, target_size=target_size, strength=strength)
                resp = {
                    "input": _encode_png_data_url(ldct),
                    "restored": _encode_png_data_url(restored),
                    "width": int(ldct.shape[1]),
                    "height": int(ldct.shape[0]),
                }
                self._send(200, json.dumps(resp).encode("utf-8"), "application/json")
            except Exception as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Launch local LDCT restoration web interface")
    parser.add_argument("--bundle", default="models/ldct_restorer_bundle.joblib", help="Path to joblib bundle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    bundle_meta = joblib.load(args.bundle).get("test_metrics", {})
    if "raw_psnr" in bundle_meta and "unet_psnr" in bundle_meta:
        badge = f"Test PSNR: {bundle_meta['raw_psnr']:.2f} → {bundle_meta['unet_psnr']:.2f} dB | SSIM: {bundle_meta['raw_ssim']:.3f} → {bundle_meta['unet_ssim']:.4f}"
    else:
        badge = "ResidualAttentionUNet (CPU)"

    model, cfg = load_bundle(args.bundle)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(model, cfg, badge))
    print(f"Serving LDCT Restoration UI at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
