import argparse
import base64
import sys
import threading
from enum import Enum
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# Carga perezosa (lazy) de modelos: la API arranca aunque falte algún fichero de modelo.
MODEL_FACTORIES = {
    ModelName.mediapipe: MediaPipe3DPoseEstimator,
    ModelName.lightweight3d: Lightweight3DPoseEstimator,
}

_models = {}
_models_lock = threading.Lock()


def get_model(model: ModelName):
    if model in _models:
        return _models[model]
    with _models_lock:
        if model not in _models:
            _models[model] = MODEL_FACTORIES[model]()
        return _models[model]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta la estimación de pose sobre una imagen y guarda el resultado en JSON."
    )
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        help="Ruta de la imagen a procesar. Si se omite, arranca el servidor FastAPI.",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=[model.value for model in ModelName],
        default=ModelName.mediapipe.value,
        help="Modelo de estimación a utilizar (por defecto: mediapipe).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results"),
        help="Carpeta o fichero .json donde guardar el resultado (por defecto: results/).",
    )
    return parser


def resolve_output_path(image_path: Path, model: ModelName, output: Path) -> Path:
    if output.suffix.lower() == ".json":
        return output
    return output / f"{image_path.stem}_{model.value}.json"


def run_cli(args: argparse.Namespace) -> int:
    model = ModelName(args.model)
    image_path = args.image

    if not image_path.is_file():
        print(f"Error: no se encontró la imagen: {image_path}", file=sys.stderr)
        return 1

    output_path = resolve_output_path(image_path, model, args.output)

    try:
        image_bytes = image_path.read_bytes()
        estimator = get_model(model)
        response = estimator.predict(image_bytes)
    except FileNotFoundError as e:
        print(f"Error: modelo no disponible: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: modelo no disponible: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: imagen inválida: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Error al leer la imagen: {e}", file=sys.stderr)
        return 1

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
    except OSError as e:
        print(f"Error al guardar el resultado: {e}", file=sys.stderr)
        return 1

    print(f"Personas detectadas: {response.persons_detected}")
    print(f"Resultado guardado en: {output_path}")
    return 0


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Human Pose Estimation API",
        "available_models": ["mediapipe", "lightweight3d"]
    }

@app.post("/pose", response_model=PoseResponse)
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
        estimator = get_model(model)
        response = estimator.predict(image_bytes)
        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {str(e)}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    if cli_args.image:
        raise SystemExit(run_cli(cli_args))

    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
