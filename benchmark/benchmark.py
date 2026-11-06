import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import MODEL_FACTORIES, ModelName
from app.schemas import PoseResponse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mide el tiempo de inferencia y el confidence promedio de un modelo de pose."
    )
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        required=True,
        help="Ruta de la imagen a procesar.",
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
        default=Path("benchmark/results"),
        help="Carpeta o fichero .json donde guardar el resultado (por defecto: benchmark/results/).",
    )
    return parser


def resolve_output_path(image_path: Path, model: ModelName, output: Path) -> Path:
    if output.suffix.lower() == ".json":
        return output
    return output / f"{image_path.stem}_{model.value}_benchmark.json"


def compute_confidence_stats(response: PoseResponse) -> Tuple[int, int, Optional[float]]:
    keypoints = [keypoint for pose in response.poses for keypoint in pose.keypoints]
    detected = [keypoint for keypoint in keypoints if keypoint.confidence > 0]

    if not detected:
        return len(keypoints), 0, None

    average = sum(keypoint.confidence for keypoint in detected) / len(detected)
    return len(keypoints), len(detected), round(average, 4)


def run_benchmark(args: argparse.Namespace) -> int:
    model = ModelName(args.model)
    image_path = args.image

    if not image_path.is_file():
        print(f"Error: no se encontró la imagen: {image_path}", file=sys.stderr)
        return 1

    output_path = resolve_output_path(image_path, model, args.output)

    try:
        image_bytes = image_path.read_bytes()
    except OSError as e:
        print(f"Error al leer la imagen: {e}", file=sys.stderr)
        return 1

    load_start = time.perf_counter()
    try:
        estimator = MODEL_FACTORIES[model]()
    except FileNotFoundError as e:
        print(f"Error: modelo no disponible: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: modelo no disponible: {e}", file=sys.stderr)
        return 1
    model_load_seconds = time.perf_counter() - load_start

    inference_start = time.perf_counter()
    try:
        response = estimator.predict(image_bytes)
    except ValueError as e:
        print(f"Error: imagen inválida: {e}", file=sys.stderr)
        return 1
    inference_seconds = time.perf_counter() - inference_start

    keypoints_total, keypoints_with_confidence, average_confidence = (
        compute_confidence_stats(response)
    )

    result = {
        "image": str(image_path),
        "model": model.value,
        "model_used": response.model_used,
        "persons_detected": response.persons_detected,
        "keypoints_total": keypoints_total,
        "keypoints_with_confidence": keypoints_with_confidence,
        "average_confidence": average_confidence,
        "model_load_seconds": round(model_load_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        print(f"Error al guardar el resultado: {e}", file=sys.stderr)
        return 1

    confidence_text = (
        f"{average_confidence:.4f}" if average_confidence is not None else "N/A"
    )
    print(f"Modelo: {response.model_used}")
    print(f"Personas detectadas: {response.persons_detected}")
    print(f"Tiempo de carga: {model_load_seconds:.4f} s")
    print(f"Tiempo de inferencia: {inference_seconds:.4f} s")
    print(f"Confidence promedio: {confidence_text}")
    print(f"Resultado guardado en: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_benchmark(build_arg_parser().parse_args()))
