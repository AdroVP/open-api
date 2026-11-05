"""Descarga los modelos necesarios para ejecutar la API en local.

Uso (desde la carpeta open-api):
    python scripts/download_models.py
"""

import os
import urllib.request

MODELS = {
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task"
    ),
    "efficientdet_lite0.tflite": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
    ),
}

LIGHTWEIGHT3D_GDRIVE_ID = "1niBUbUecPhKt3GyeDNukobL4OQ3jqssH"
LIGHTWEIGHT3D_PATH = "human-pose-estimation-3d.pth"


def download_url(url: str, destination: str) -> None:
    if os.path.exists(destination):
        print(f"[OK] Ya existe {destination}")
        return
    print(f"[..] Descargando {destination}")
    urllib.request.urlretrieve(url, destination)
    print(f"[OK] {destination}")


def download_lightweight3d() -> None:
    if os.path.exists(LIGHTWEIGHT3D_PATH):
        print(f"[OK] Ya existe {LIGHTWEIGHT3D_PATH}")
        return
    try:
        import gdown
    except ImportError:
        raise SystemExit("Falta 'gdown'. Instala las dependencias: pip install -r requirements.txt")

    print(f"[..] Descargando {LIGHTWEIGHT3D_PATH}")
    gdown.download(id=LIGHTWEIGHT3D_GDRIVE_ID, output=LIGHTWEIGHT3D_PATH, quiet=False)
    print(f"[OK] {LIGHTWEIGHT3D_PATH}")


if __name__ == "__main__":
    for name, url in MODELS.items():
        download_url(url, name)
    download_lightweight3d()
