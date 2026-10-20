import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

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

def test_pose_endpoint_no_image():
    response = client.post("/pose?model=mediapipe")
    assert response.status_code == 400