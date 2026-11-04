from typing import List, Optional
from pydantic import BaseModel, Field


class Coordinate3D(BaseModel):
    x: float = Field(..., description="Coordenada X en metros (espacio del mundo/cámara)")
    y: float = Field(..., description="Coordenada Y en metros (espacio del mundo/cámara)")
    z: float = Field(..., description="Coordenada Z en metros (espacio del mundo/cámara)")


class Keypoint2D(BaseModel):
    name: str = Field(..., description="Nombre de la articulación/keypoint")
    x: float = Field(..., description="Coordenada X normalizada (0.0 - 1.0) o en píxeles")
    y: float = Field(..., description="Coordenada Y normalizada (0.0 - 1.0) o en píxeles")
    confidence: float = Field(..., description="Grado de confianza de la detección (0.0 - 1.0)")


class Keypoint3D(Keypoint2D):
    z: float = Field(..., description="Coordenada Z de profundidad (metros)")
    coordinates_3d_meters: Optional[Coordinate3D] = Field(
        default=None,
        description="Coordenadas 3D completas en el espacio del mundo/cámara (metros)",
    )


class PersonPose(BaseModel):
    person_id: int = Field(..., description="Identificador único de la persona detectada")
    keypoints: List[Keypoint3D] = Field(..., description="Lista de articulaciones 2D/3D")


class PoseResponse(BaseModel):
    model_used: str = Field(..., description="Modelo utilizado para la inferencia")
    persons_detected: int = Field(..., description="Cantidad de personas detectadas")
    poses: List[PersonPose] = Field(..., description="Lista de poses detectadas")
