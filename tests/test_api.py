import importlib.util
import io
import os

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import PoseResponse


client = TestClient(app)

POSELANDMARKER = os.path.exists("pose_landmarker_full.task")
LIGHTWEIGHT3D_WEIGHTS = os.path.exists("human-pose-estimation-3d.pth")
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

requires_mediapipe = pytest.mark.skipif(
    not POSELANDMARKER,
    reason="Falta pose_landmarker_full.task en el directorio de trabajo",
)

requires_lightweight3d = pytest.mark.skipif(
    not (LIGHTWEIGHT3D_WEIGHTS and TORCH_AVAILABLE),
    reason="Faltan los pesos de Lightweight 3D o PyTorch",
)


def create_test_image():
    file = io.BytesIO()
    image = Image.new('RGB', (100, 100), color='red')
    image.save(file, 'jpeg')
    file.seek(0)
    return file


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_pose_endpoint_no_image():
    response = client.post("/pose?model=mediapipe")
    assert response.status_code == 400


@requires_mediapipe
def test_pose_endpoint_multipart():
    img_bytes = create_test_image()
    response = client.post(
        "/pose?model=mediapipe",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "model_used" in json_data
    assert "persons_detected" in json_data
    assert "poses" in json_data
    assert json_data["persons_detected"] == len(json_data["poses"])


@requires_lightweight3d
def test_pose_endpoint_lightweight3d():
    img_bytes = create_test_image()
    response = client.post(
        "/pose?model=lightweight3d",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["model_used"] == "Lightweight 3D Pose Estimation"
    assert json_data["persons_detected"] == len(json_data["poses"])


def test_pose_response_allows_no_persons():
    response = PoseResponse(
        model_used="test",
        persons_detected=0,
        poses=[],
    )
    assert response.persons_detected == 0


def test_lightweight3d_keypoint_order():
    from app.models.lightweight_3d import Lightweight3DPoseEstimator

    assert len(Lightweight3DPoseEstimator.KEYPOINT_NAMES) == 19
    assert Lightweight3DPoseEstimator.KEYPOINT_NAMES[0] == "Nose"
    assert Lightweight3DPoseEstimator.KEYPOINT_NAMES[1] == "Neck"
    assert Lightweight3DPoseEstimator.KEYPOINT_NAMES[3] == "L_Shoulder"
    assert Lightweight3DPoseEstimator.KEYPOINT_NAMES[9] == "R_Shoulder"
