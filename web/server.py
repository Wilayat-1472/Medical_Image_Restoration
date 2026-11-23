"""Local-only image restoration UI, backed by the project's real PyTorch model."""
import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit, parse_qs
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from data.io import unit_float
from inference.confidence import ConfidenceScorer
from inference.failsafe import FailSafe
from inference.restore import RestorationPipeline
from utils.checkpoint import load_model
from evaluation.detail import compare_detail

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name('static')
DEFAULT_CHECKPOINT = ROOT / 'checkpoints' / 'colab_t4' / 'light_unet_best.pt'
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
ALLOWED_FORMATS = {'PNG', 'JPEG', 'TIFF', 'BMP'}
LOGGER = logging.getLogger(__name__)


class ImageValidationError(ValueError):
    pass


def decode_upload(content):
    """Inspect dimensions before decoding; retain 16-bit grayscale intensities."""
    if not content:
        raise ImageValidationError('Choose an image before continuing.')
    if len(content) > MAX_UPLOAD_BYTES:
        raise ImageValidationError('This file is too large. Choose an image under 20 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in ALLOWED_FORMATS:
                    raise ImageValidationError('Choose a PNG, JPG, TIFF or BMP image.')
                if getattr(source, 'n_frames', 1) != 1:
                    raise ImageValidationError('Choose a single image. Multi-page TIFF files must be converted first.')
                width, height = source.size
                if width * height > MAX_PIXELS or max(width, height) > 4096:
                    raise ImageValidationError('Choose an image no larger than 4096 × 4096 pixels.')
                image = ImageOps.exif_transpose(source)
                if image.mode not in ('L', 'I;16', 'I;16L', 'I;16B', 'RGB', 'RGBA'):
                    image = image.convert('RGB')
                # Composite transparency for a faithful, visible preview.
                if image.mode == 'RGBA':
                    background = Image.new('RGBA', image.size, 'white')
                    image = Image.alpha_composite(background, image).convert('RGB')
                array = np.array(image)
                if array.dtype.kind == 'u' and array.dtype.itemsize == 2:
                    array = array.astype(np.uint16)
                array = unit_float(array)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError('This file could not be read. Choose a valid PNG, JPG, TIFF or BMP image.') from exc
    validation = FailSafe().validate_input(array)
    if not validation['valid']:
        if min(array.shape[:2]) < 32:
            raise ImageValidationError('This image is too small. Use an image at least 32 × 32 pixels.')
        raise ImageValidationError('This image has no usable detail. Choose an image with a visible range of intensities.')
    return array


def png_data_url(image):
    array = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


class RestorationService:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f'Trained model not found: {checkpoint}. Supply its location with --checkpoint.')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, self.config = load_model(checkpoint, device=device)
        self.pipeline = RestorationPipeline(model, device=device, target_size=self.config['data']['target_size'])
        threshold = self.config.get('inference', {}).get('confidence_threshold', 0.6)
        self.scorer = ConfidenceScorer(threshold)
        self.failsafe = FailSafe(threshold)
        self.lock = threading.Lock()

    def status(self):
        return {'ready': True, 'model': 'Chest X-ray restoration',
                'processing_size': self.pipeline.target_size,
                'max_upload_mb': MAX_UPLOAD_BYTES // (1024 * 1024),
                'default_strength': self.pipeline.strength, 'detail_preserving': True}

    def process(self, content, *, preview=False, strength=None):
        image = decode_upload(content)
        height, width = image.shape[:2]
        original = png_data_url(image)
        if preview:
            return {'original': original, 'width': width, 'height': height}
        # A single cached model is shared safely by concurrent browser requests.
        with self.lock:
            start = time.perf_counter()
            restored = self.pipeline.restore(image, strength=strength)['restored']
            elapsed = (time.perf_counter() - start) * 1000
        confidence = self.scorer.score(image, restored)
        quality = self.failsafe.check_restoration_quality(restored, confidence['overall_confidence'])
        detail = compare_detail(image, restored)
        if detail['weaker_edges']:
            quality['passes'] = False
            quality['warnings'].append('The result has substantially weaker edges. Lower restoration strength and review fine details.')
        return {'original': original, 'restored': png_data_url(restored),
                'width': width, 'height': height, 'elapsed_ms': round(elapsed, 1),
                'review_recommended': not quality['passes'],
                'warnings': quality['warnings'], 'detail_check': detail,
                'strength': self.pipeline.strength if strength is None else strength,
                'processing_size': self.pipeline.target_size}


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def send_content(self, status, content, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, status, payload):
            self.send_content(status, json.dumps(payload).encode('utf-8'), 'application/json; charset=utf-8')

        def allowed_request(self):
            host = self.headers.get('Host', '')
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if host not in allowed:
                self.send_json(HTTPStatus.FORBIDDEN, {'error': 'Open this app using its localhost address.'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://{name}' for name in allowed}:
                self.send_json(HTTPStatus.FORBIDDEN, {'error': 'Open this app directly to upload an image.'})
                return False
            return True

        def do_GET(self):
            if not self.allowed_request():
                return
            path = urlsplit(self.path).path
            if path == '/api/status':
                self.send_json(HTTPStatus.OK, service.status())
                return
            files = {'/': ('index.html', 'text/html; charset=utf-8'),
                     '/app.css': ('app.css', 'text/css; charset=utf-8'),
                     '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                     '/favicon.svg': ('favicon.svg', 'image/svg+xml'),
                     '/sample.png': ('sample.png', 'image/png')}
            if path not in files:
                self.send_json(HTTPStatus.NOT_FOUND, {'error': 'Page not found.'})
                return
            name, content_type = files[path]
            self.send_content(HTTPStatus.OK, (STATIC / name).read_bytes(), content_type)

        def do_POST(self):
            if not self.allowed_request():
                return
            path = urlsplit(self.path).path
            if path not in ('/api/preview', '/api/restore'):
                self.send_json(HTTPStatus.NOT_FOUND, {'error': 'Page not found.'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.send_json(HTTPStatus.BAD_REQUEST, {'error': 'The upload could not be read.'})
                return
            if length < 1 or length > MAX_UPLOAD_BYTES:
                self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length > MAX_UPLOAD_BYTES else HTTPStatus.BAD_REQUEST,
                               {'error': 'Choose an image under 20 MB.'})
                return
            try:
                content = self.rfile.read(length)
                if len(content) != length:
                    raise ImageValidationError('The upload was interrupted. Please try again.')
                strength = parse_qs(urlsplit(self.path).query).get('strength', [None])[0]
                if strength is not None:
                    try:
                        strength = RestorationPipeline._validate_strength(strength)
                    except (ValueError, TypeError) as exc:
                        raise ImageValidationError('Restoration strength must be between 0 and 1.') from exc
                payload = service.process(content, preview=path == '/api/preview', strength=strength)
            except ImageValidationError as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            except Exception:
                LOGGER.exception('Restoration request failed')
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                               {'error': 'The image could not be restored. Please try another image.'})
                return
            self.send_json(HTTPStatus.OK, payload)

        def log_message(self, format, *args):
            LOGGER.info('%s %s', self.address_string(), format % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description='Open the local medical image restoration interface')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    torch.set_num_threads(2)
    try:
        service = RestorationService(args.checkpoint)
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(service))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Could not start the interface: {exc}\n')
    print(f'\nMedical image restoration is ready: http://127.0.0.1:{server.server_port}\n', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
