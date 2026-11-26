"""Tests for ResidualAttentionUNet, joblib bundle serialization, and predict.py."""
import cv2
import joblib
import numpy as np
import torch

from predict import load_bundle, load_ldct_slice, restore_slice
from src.model import ResidualAttentionUNet


def test_residual_attention_unet_shapes_and_grad():
    model = ResidualAttentionUNet(in_channels=1, out_channels=1, base_c=16)
    x = torch.rand(2, 1, 67, 73)
    y = model(x)
    assert y.shape == x.shape
    assert 0.0 <= y.min() <= y.max() <= 1.0
    loss = (y - torch.rand_like(y)).abs().mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_joblib_bundle_and_predict_cpu(tmp_path):
    cfg = {"in_channels": 1, "out_channels": 1, "base_c": 16, "image_size": 64}
    model = ResidualAttentionUNet(in_channels=1, out_channels=1, base_c=16).eval()
    bundle_path = tmp_path / "ldct_restorer_bundle.joblib"
    joblib.dump(
        {
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": cfg,
        },
        bundle_path,
        compress=3,
    )

    loaded_model, loaded_cfg = load_bundle(bundle_path)
    assert loaded_cfg == cfg

    inp_path = tmp_path / "ldct.png"
    rng = np.random.default_rng(42)
    img_u8 = (rng.random((80, 96)) * 255).astype(np.uint8)
    cv2.imwrite(str(inp_path), img_u8)

    ldct = load_ldct_slice(inp_path)
    assert ldct.shape == (80, 96)
    restored = restore_slice(loaded_model, ldct, target_size=loaded_cfg["image_size"], strength=0.8)
    assert restored.shape == (80, 96)
    assert 0.0 <= restored.min() <= restored.max() <= 1.0
    np.testing.assert_allclose(restore_slice(loaded_model, ldct, target_size=64, strength=0.0), ldct, atol=1e-6)


def test_web_app_endpoints(tmp_path):
    import base64
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from app import create_handler

    model = ResidualAttentionUNet(in_channels=1, out_channels=1, base_c=16).eval()
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(model, {"image_size": 32}, "Test Badge"))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/health") as r:
            assert r.status == 200 and json.loads(r.read())["status"] == "ok"
        with urlopen(f"http://127.0.0.1:{port}/") as r:
            assert b"Test Badge" in r.read()

        img = (np.random.default_rng(1).random((32, 32)) * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        body = json.dumps({
            "filename": "slice.png",
            "data": base64.b64encode(buf.tobytes()).decode("ascii"),
            "strength": 0.9,
        }).encode("utf-8")
        req = Request(f"http://127.0.0.1:{port}/api/restore", data=body, headers={"Content-Type": "application/json"})
        with urlopen(req) as r:
            res = json.loads(r.read())
            assert res["width"] == 32 and res["height"] == 32
            assert res["restored"].startswith("data:image/png;base64,")
    finally:
        server.shutdown()
        server.server_close()
