MODALITY_INFO = {
    "ct": {
        "channels": 1,
        "intensity_range": (-1000, 3000),
        "default_window": (40, 80),
        "description": "Computed Tomography",
    },
    "mri": {
        "channels": 1,
        "intensity_range": (0, 4096),
        "default_window": None,
        "description": "Magnetic Resonance Imaging",
    },
    "xray": {
        "channels": 1,
        "intensity_range": (0, 65535),
        "default_window": None,
        "description": "X-Ray Radiography",
    },
    "retinal": {
        "channels": 3,
        "intensity_range": (0, 255),
        "default_window": None,
        "description": "Retinal Fundus Photography",
    },
    "ultrasound": {
        "channels": 1,
        "intensity_range": (0, 255),
        "default_window": None,
        "description": "Ultrasound Imaging",
    },
    "histopathology": {
        "channels": 3,
        "intensity_range": (0, 255),
        "default_window": None,
        "description": "Histopathology Slide",
    },
}

DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD = [0.5, 0.5, 0.5]

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".nii", ".nii.gz"}
SUPPORTED_MODALITIES = list(MODALITY_INFO.keys())
