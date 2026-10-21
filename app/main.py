import base64
from enum import Enum
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from app.schemas import PoseResponse
from app.models.mediapipe_pose import MediaPipe3DPoseEstimator
from app.models.lightweight_3d import Lightweight3DPoseEstimator

class ModelName(str, Enum):
    mediapipe = "mediapipe"
    lightweight3d = "lightweight3d"

app = FastAPI(
    title="Human Pose Estimation REST API",
    description="Servicio de inferencia de pose humana 2D/3D con licencias permisivas (Apache 2.0 / MIT)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Carga e inicialización de modelos
models = {
    ModelName.mediapipe: MediaPipe3DPoseEstimator(),
    ModelName.lightweight3d: Lightweight3DPoseEstimator()
}

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Human Pose Estimation API",
        "available_models": ["mediapipe", "lightweight3d"]
    }

@app.post("/pose")
async def estimate_pose(
    model: ModelName = Query(ModelName.mediapipe, description="Modelo de estimación a utilizar"),
    file: Optional[UploadFile] = File(None, description="Imagen codificada en multipart/form-data"),
    image_base64: Optional[str] = Form(None, description="Imagen codificada en cadena Base64")
):
    """
    Procesa una imagen recibida vía **multipart/form-data** o **base64** y devuelve los keypoints con coordenadas (x, y, z), confianza y nombre de la articulación.
    """
    image_bytes = None

    if file:
        image_bytes = await file.read()
    elif image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 inválido: {str(e)}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Debe proporcionar una imagen mediante 'file' (multipart/form-data) o 'image_base64' (Form)."
        )

    try:
        estimator = models[model]
        response = estimator.predict(image_bytes)
        return {"keypoints": response}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")