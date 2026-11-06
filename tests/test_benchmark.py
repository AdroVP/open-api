from pathlib import Path

from app.main import ModelName
from app.schemas import Keypoint3D, PersonPose, PoseResponse
from benchmark.benchmark import compute_confidence_stats, resolve_output_path


def make_keypoint(name: str, confidence: float) -> Keypoint3D:
    return Keypoint3D(name=name, x=0.0, y=0.0, z=0.0, confidence=confidence)


def test_resolve_output_path_default_directory():
    output = resolve_output_path(
        Path("assets/Imagen_1.png"),
        ModelName.mediapipe,
        Path("benchmark/results"),
    )
    assert output == Path("benchmark/results/Imagen_1_mediapipe_benchmark.json")


def test_resolve_output_path_explicit_json_file():
    output = resolve_output_path(
        Path("assets/Imagen_1.png"),
        ModelName.lightweight3d,
        Path("custom/result.json"),
    )
    assert output == Path("custom/result.json")


def test_compute_confidence_stats_ignores_not_detected_keypoints():
    response = PoseResponse(
        model_used="test",
        persons_detected=2,
        poses=[
            PersonPose(
                person_id=1,
                keypoints=[
                    make_keypoint("nose", 0.5),
                    make_keypoint("neck", 0.0),
                ],
            ),
            PersonPose(
                person_id=2,
                keypoints=[
                    make_keypoint("nose", 1.0),
                    make_keypoint("neck", 0.5),
                ],
            ),
        ],
    )

    total, detected, average = compute_confidence_stats(response)

    assert total == 4
    assert detected == 3
    assert average == round((0.5 + 1.0 + 0.5) / 3, 4)


def test_compute_confidence_stats_without_detected_keypoints():
    response = PoseResponse(
        model_used="test",
        persons_detected=1,
        poses=[
            PersonPose(
                person_id=1,
                keypoints=[make_keypoint("nose", 0.0)],
            )
        ],
    )

    total, detected, average = compute_confidence_stats(response)

    assert total == 1
    assert detected == 0
    assert average is None
