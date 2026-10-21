import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class MediaPipe3DPoseEstimator:
    def __init__(self, model_path: str = "pose_landmarker_full.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

        # 1. Configurar las opciones del motor Tasks
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 2. Crear el detector de pose
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def predict(self, image_bytes: bytes):
        # Convertir bytes a matriz OpenCV (BGR)
        nparr = np.frombuffer(image_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        # Convertir de BGR a RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # Crear objeto Image de MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        # Realizar la detección
        detection_result = self.detector.detect(mp_image)

        keypoints = []

        # Verificar si se detectaron poses
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            landmarks_2d = detection_result.pose_landmarks[0]
            landmarks_3d = detection_result.pose_world_landmarks[0] if detection_result.pose_world_landmarks else []

            # Mapeo de nombres oficiales de los 33 puntos de MediaPipe
            landmark_names = [
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

            for idx, lm in enumerate(landmarks_2d):
                name = landmark_names[idx] if idx < len(landmark_names) else f"joint_{idx}"

                # Coordenadas 3D del mundo real (en metros)
                world_3d = None
                if idx < len(landmarks_3d):
                    w_lm = landmarks_3d[idx]
                    world_3d = {
                        "x": round(float(w_lm.x), 4),
                        "y": round(float(w_lm.y), 4),
                        "z": round(float(w_lm.z), 4)
                    }

                keypoints.append({
                    "name": name,
                    "x": round(float(lm.x), 4),  # Normalizado [0.0, 1.0]
                    "y": round(float(lm.y), 4),  # Normalizado [0.0, 1.0]
                    "confidence": round(float(getattr(lm, 'visibility', 1.0)), 4),
                    "coordinates_3d_meters": world_3d
                })

        return keypoints