"""Image upload contracts and the HTTP paths used by the browser interface."""
import base64
import io
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from models.factory import create_model
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from web.server import (MAX_UPLOAD_BYTES, ImageValidationError, RestorationService,
                        decode_upload, make_handler)


def image_bytes(array=None, format='PNG', **kwargs):
    if array is None:
        array = np.random.default_rng(1).integers(0, 256, (40, 48, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=format, **kwargs)
    return buffer.getvalue()


@pytest.fixture(scope='module')
def service(tmp_path_factory):
    path = tmp_path_factory.mktemp('web') / 'model.pt'
    config = load_config('configs/cpu_config.yaml')
    config['model']['light_unet'].update(base_filters=8, depth=2)
    config['data']['target_size'] = 32
    save_checkpoint(path, create_model(config), config)
    return RestorationService(path)


@pytest.mark.parametrize('format', ['PNG', 'JPEG', 'TIFF', 'BMP'])
def test_upload_supported_formats(format):
    array = decode_upload(image_bytes(format=format))
    assert array.shape == (40, 48, 3)
    assert array.dtype == np.float32
    assert 0 <= array.min() <= array.max() <= 1


def test_uint16_and_exif_orientation():
    values = np.arange(40 * 48, dtype=np.uint16).reshape(40, 48) * 30
    array = decode_upload(image_bytes(values))
    assert array.shape == (40, 48, 1)
    assert array.max() == pytest.approx(values.max() / 65535)
    exif = Image.Exif()
    exif[274] = 6
    assert decode_upload(image_bytes(format='JPEG', exif=exif)).shape == (48, 40, 3)


@pytest.mark.parametrize('content', [b'', b'not an image', image_bytes(np.zeros((40, 48), dtype=np.uint8)),
                                    image_bytes(np.zeros((10, 10), dtype=np.uint8)),
                                    image_bytes(np.zeros((32, 4097), dtype=np.uint8)),
                                    image_bytes(format='GIF')])
def test_reject_unusable_images(content):
    with pytest.raises(ImageValidationError):
        decode_upload(content)


def test_reject_multipage_tiff():
    first = Image.fromarray(np.arange(40 * 48, dtype=np.uint16).reshape(40, 48))
    buffer = io.BytesIO()
    first.save(buffer, format='TIFF', save_all=True, append_images=[first])
    with pytest.raises(ImageValidationError, match='Multi-page'):
        decode_upload(buffer.getvalue())


def test_real_model_response(service):
    content = image_bytes()
    preview = service.process(content, preview=True)
    assert preview['width'] == 48 and preview['height'] == 40
    assert 'restored' not in preview
    result = service.process(content)
    assert result['elapsed_ms'] >= 0
    assert result['processing_size'] == 32
    assert isinstance(result['review_recommended'], bool)
    png = base64.b64decode(result['restored'].split(',', 1)[1])
    with Image.open(io.BytesIO(png)) as image:
        assert image.size == (48, 40) and image.format == 'PNG'


class RequestSocket:
    """Run a real BaseHTTPRequestHandler without binding a network socket."""
    def __init__(self, request):
        self.request = io.BytesIO(request)
        self.response = bytearray()

    def makefile(self, *args):
        return self.request

    def settimeout(self, timeout):
        pass

    def sendall(self, data):
        self.response.extend(data)


def request(service, path, body=None, headers=None):
    headers = {'Host': '127.0.0.1:7860', **(headers or {})}
    if body is not None:
        headers.setdefault('Content-Length', str(len(body)))
    head = f'{"POST" if body is not None else "GET"} {path} HTTP/1.0\r\n'
    head += ''.join(f'{key}: {value}\r\n' for key, value in headers.items()) + '\r\n'
    connection = RequestSocket(head.encode() + (body or b''))
    make_handler(service)(connection, ('127.0.0.1', 1234), SimpleNamespace(server_port=7860))
    header, content = bytes(connection.response).split(b'\r\n\r\n', 1)
    return int(header.split()[1]), header.decode(), content


def test_http_pages_and_api(service):
    for path in ['/', '/app.css', '/app.js', '/favicon.svg', '/sample.png']:
        status, headers, body = request(service, path)
        assert status == 200 and body
        assert 'Cache-Control: no-store' in headers
    status, _, body = request(service, '/api/status')
    assert status == 200 and json.loads(body)['ready']
    status, _, body = request(service, '/api/restore', body=image_bytes())
    assert status == 200 and json.loads(body)['restored'].startswith('data:image/png;base64,')
    status, _, body = request(service, '/api/preview', body=image_bytes())
    assert status == 200 and json.loads(body)['width'] == 48


def test_http_upload_errors_and_local_access(service):
    assert request(service, '/api/restore', body=b'invalid')[0] == 400
    assert request(service, '/api/restore', body=b'x', headers={'Content-Length': str(MAX_UPLOAD_BYTES + 1)})[0] == 413
    assert request(service, '/api/restore', body=b'x', headers={'Content-Length': 'invalid'})[0] == 400
    assert request(service, '/api/restore', body=b'x', headers={'Origin': 'https://example.com'})[0] == 403
    assert request(service, '/', headers={'Host': 'example.com:7860'})[0] == 403
    assert request(service, '/../checkpoints/light_unet_best.pt')[0] == 404
    assert request(service, '/api/unknown', body=b'x')[0] == 404


def test_strength_api_and_original_preservation(service):
    status, _, body = request(service, '/api/restore?strength=0', body=image_bytes())
    result = json.loads(body)
    assert status == 200 and result['strength'] == 0
    assert result['original'] == result['restored']
    for value in ['-1', '2', 'nan', 'broken']:
        status, _, _ = request(service, f'/api/restore?strength={value}', body=image_bytes())
        assert status == 400
