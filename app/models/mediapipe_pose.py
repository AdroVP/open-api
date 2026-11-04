import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

from app.schemas import Coordinate3D, Keypoint3D, PersonPose, PoseResponse


class MediaPipe3DPoseEstimator:
    """
    Estimación de pose multi-persona con MediaPipe Tasks.

    El modelo PoseLandmarker por defecto (BlazePose) es single-person, por lo que
    se usa una estrategia top-down: un ObjectDetector localiza a cada persona,
    se recorta su bounding box y se ejecuta PoseLandmarker sobre cada recorte.
    """

    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    def __init__(
        self,
        model_path: str = "pose_landmarker_full.task",
        detector_path: str = "efficientdet_lite0.tflite",
        max_persons: int = 10,
        person_score_threshold: float = 0.3,
        bbox_margin: float = 0.25,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

        self.bbox_margin = bbox_margin

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)

        self.object_detector = None
        if os.path.exists(detector_path):
            object_options = vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=detector_path),
                running_mode=vision.RunningMode.IMAGE,
                score_threshold=person_score_threshold,
                max_results=max_persons,
                category_allowlist=["person"],
            )
            self.object_detector = vision.ObjectDetector.create_from_options(object_options)
        else:
            print(
                f"[WARNING] No se encontró '{detector_path}'. "
                "MediaPipe funcionará en modo single-person."
            )

    def _build_person_pose(
        self,
        landmarks_2d,
        landmarks_3d,
        person_id: int,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        crop_w: int = None,
        crop_h: int = None,
        img_w: int = None,
        img_h: int = None,
    ) -> PersonPose:
        keypoints = []

        for idx, lm in enumerate(landmarks_2d):
            name = self.LANDMARK_NAMES[idx] if idx < len(self.LANDMARK_NAMES) else f"joint_{idx}"

            x = float(lm.x)
            y = float(lm.y)
            if crop_w is not None and img_w is not None:
                x = (x * crop_w + offset_x) / img_w
            if crop_h is not None and img_h is not None:
                y = (y * crop_h + offset_y) / img_h

            world_3d = None
            z = float(lm.z)
            if landmarks_3d and idx < len(landmarks_3d):
                w_lm = landmarks_3d[idx]
                world_3d = Coordinate3D(
                    x=round(float(w_lm.x), 4),
                    y=round(float(w_lm.y), 4),
                    z=round(float(w_lm.z), 4),
                )
                z = world_3d.z

            keypoints.append(
                Keypoint3D(
                    name=name,
                    x=round(x, 4),
                    y=round(y, 4),
                    z=round(z, 4),
                    confidence=round(float(getattr(lm, "visibility", 1.0)), 4),
                    coordinates_3d_meters=world_3d,
                )
            )

        return PersonPose(person_id=person_id, keypoints=keypoints)

    def _detect_single(self, rgb_img, offset_x, offset_y, crop_w, crop_h, img_w, img_h, person_id):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_img),
        )
        result = self.pose_detector.detect(mp_image)

        if not result.pose_landmarks:
            return None

        world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
        return self._build_person_pose(
            result.pose_landmarks[0],
            world_landmarks,
            person_id=person_id,
            offset_x=offset_x,
            offset_y=offset_y,
            crop_w=crop_w,
            crop_h=crop_h,
            img_w=img_w,
            img_h=img_h,
        )

    def _bbox_to_pixels(self, bbox, img_w: int, img_h: int):
        if (
            bbox.origin_x <= 1
            and bbox.origin_y <= 1
            and bbox.width <= 1
            and bbox.height <= 1
        ):
            return (
                bbox.origin_x * img_w,
                bbox.origin_y * img_h,
                bbox.width * img_w,
                bbox.height * img_h,
            )
        return float(bbox.origin_x), float(bbox.origin_y), float(bbox.width), float(bbox.height)

    def predict(self, image_bytes: bytes) -> PoseResponse:
        nparr = np.frombuffer(image_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb_img.shape[:2]

        poses = []

        if self.object_detector is not None:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_img),
            )
            detection_result = self.object_detector.detect(mp_image)

            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x, y, bw, bh = self._bbox_to_pixels(bbox, img_w, img_h)

                pad_x = self.bbox_margin * bw
                pad_y = self.bbox_margin * bh
                x0 = max(0, int(round(x - pad_x)))
                y0 = max(0, int(round(y - pad_y)))
                x1 = min(img_w, int(round(x + bw + pad_x)))
                y1 = min(img_h, int(round(y + bh + pad_y)))

                if x1 - x0 < 10 or y1 - y0 < 10:
                    continue

                crop = rgb_img[y0:y1, x0:x1]
                person_pose = self._detect_single(
                    crop,
                    offset_x=x0,
                    offset_y=y0,
                    crop_w=x1 - x0,
                    crop_h=y1 - y0,
                    img_w=img_w,
                    img_h=img_h,
                    person_id=len(poses) + 1,
                )
                if person_pose is not None:
                    poses.append(person_pose)

        # Fallback: si no se detectó ninguna persona, probar la imagen completa
        # (preserva el comportamiento single-person cuando el detector falla).
        if not poses:
            person_pose = self._detect_single(
                rgb_img,
                offset_x=0,
                offset_y=0,
                crop_w=img_w,
                crop_h=img_h,
                img_w=img_w,
                img_h=img_h,
                person_id=1,
            )
            if person_pose is not None:
                poses.append(person_pose)

        return PoseResponse(
            model_used="MediaPipe Pose Landmarker (top-down)",
            persons_detected=len(poses),
            poses=poses,
        )
