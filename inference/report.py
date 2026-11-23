import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from matplotlib import pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


def generate_report(
    degraded_path: str,
    metrics: Dict[str, float],
    confidence: Dict[str, Any],
    restoration_time_ms: Optional[float] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    report = {
        "report_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "input_file": degraded_path,
        "model_name": model_name or "unknown",
        "metrics": metrics,
        "confidence": confidence,
        "restoration_time_ms": restoration_time_ms,
        "status": "heuristic_pass" if confidence.get("is_reliable", False) else "needs_review",
    }

    if metrics:
        report["assessment"] = _generate_assessment(metrics)

    return report


def _generate_assessment(metrics: Dict[str, float]) -> str:
    parts = []
    if "psnr" in metrics:
        psnr = metrics["psnr"]
        if psnr > 35:
            parts.append("Excellent restoration quality")
        elif psnr > 28:
            parts.append("Good restoration quality")
        elif psnr > 22:
            parts.append("Moderate restoration quality")
        else:
            parts.append("Poor restoration quality")
    if "ssim" in metrics:
        ssim = metrics["ssim"]
        if ssim > 0.95:
            parts.append("excellent structural preservation")
        elif ssim > 0.85:
            parts.append("good structural preservation")
        elif ssim > 0.70:
            parts.append("moderate structural preservation")
        else:
            parts.append("poor structural preservation")
    return ". ".join(parts) + "."


def save_report(
    report: Dict[str, Any],
    save_path: str,
    format: str = "json",
    degraded_image: Optional[np.ndarray] = None,
    restored_image: Optional[np.ndarray] = None,
) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        path = save_path if save_path.endswith(".json") else save_path + ".json"
        serializable = _make_serializable(report)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        return path

    elif format == "html" and _HAS_PLT:
        path = save_path if save_path.endswith(".html") else save_path + ".html"
        _save_html_report(report, path, degraded_image, restored_image)
        return path

    else:
        path = save_path if save_path.endswith(".json") else save_path + ".json"
        serializable = _make_serializable(report)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        return path


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_html_report(
    report: Dict[str, Any],
    path: str,
    degraded: Optional[np.ndarray] = None,
    restored: Optional[np.ndarray] = None,
):
    image_html = ""
    if degraded is not None and restored is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(degraded.squeeze(), cmap="gray")
        axes[0].set_title("Degraded")
        axes[0].axis("off")
        axes[1].imshow(restored.squeeze(), cmap="gray")
        axes[1].set_title("Restored")
        axes[1].axis("off")
        plt.tight_layout()
        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        image_html = f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;"/>'
        plt.close()

    metrics_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        for k, v in report.get("metrics", {}).items()
    )
    conf_items = "".join(
        f"<li>{k}: {v}</li>"
        for k, v in report.get("confidence", {}).items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head><title>Restoration Report - {report['report_id']}</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
.container {{ max-width: 800px; margin: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ddd; padding: 8px; }}
.status-heuristic_pass {{ color: green; }}
.status-needs_review {{ color: red; }}
</style></head>
<body>
<div class="container">
<h1>Medical Image Restoration Report</h1>
<p>Report ID: {report['report_id']}</p>
<p>Timestamp: {report['timestamp']}</p>
<p>Input: {report['input_file']}</p>
<p>Model: {report['model_name']}</p>
<p>Status: <span class="status-{report['status']}">{report['status']}</span></p>
{image_html}
<h2>Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{metrics_rows}</table>
<h2>Confidence</h2>
<ul>{conf_items}</ul>
<p>Restoration time: {report.get('restoration_time_ms', 'N/A')} ms</p>
<p>Assessment: {report.get('assessment', 'N/A')}</p>
</div></body></html>"""
    with open(path, "w") as f:
        f.write(html)
