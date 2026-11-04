import os

import cv2
import numpy as np

from app.schemas import Coordinate3D, Keypoint3D, PersonPose, PoseResponse


class Lightweight3DPoseEstimator:
    """
    Estimación real de pose 3D multi-persona basada en el modelo
    Lightweight 3D Human Pose Estimation (Apache 2.0).

    Fuente: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch
    """

    KEYPOINT_NAMES = [
        "Nose", "Neck", "Body_Center", "L_Shoulder", "L_Elbow", "L_Wrist",
        "L_Hip", "L_Knee", "L_Ankle", "R_Shoulder", "R_Elbow", "R_Wrist",
        "R_Hip", "R_Knee", "R_Ankle", "R_Eye", "L_Eye", "R_Ear", "L_Ear",
    ]

    # El modelo expresa las distancias asumiendo una altura media de 180 cm.
    MODEL_UNITS_TO_METERS = 0.01

    def __init__(
        self,
        model_path: str = "human-pose-estimation-3d.pth",
        device: str = "cpu",
        base_height: int = 256,
        stride: int = 8,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "PyTorch no está instalado. Instala las dependencias: pip install -r requirements.txt"
            ) from e

        from app.models.lightweight3d.load_state import load_state
        from app.models.lightweight3d.with_mobilenet import PoseEstimationWithMobileNet

        self._torch = torch
        self.base_height = base_height
        self.stride = stride
        self.img_mean = np.array([128, 128, 128], dtype=np.float32)
        self.img_scale = np.float32(1 / 255)

        if device != "cpu" and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        net = PoseEstimationWithMobileNet()
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location="cpu")
        load_state(net, checkpoint)
        net = net.to(self.device)
        net.eval()

        self.net = net

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return (img.astype(np.float32) - self.img_mean) * self.img_scale

    def _infer(self, img: np.ndarray):
        torch = self._torch
        normalized_img = self._normalize(img)
        data = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features, heatmaps, pafs = self.net(data)

        return (
            features[-1].squeeze().cpu().numpy(),
            heatmaps[-1].squeeze().cpu().numpy(),
            pafs[-1].squeeze().cpu().numpy(),
        )

    def predict(self, image_bytes: bytes) -> PoseResponse:
        from app.models.lightweight3d.parse_poses import parse_poses

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("La imagen no tiene dimensiones válidas.")

        input_scale = self.base_height / h
        scaled_img = cv2.resize(img, dsize=None, fx=input_scale, fy=input_scale)
        valid_width = scaled_img.shape[1] - (scaled_img.shape[1] % self.stride)
        if valid_width <= 0:
            raise ValueError("La imagen es demasiado estrecha para el modelo.")

        scaled_img = scaled_img[:, 0:valid_width]

        inference_result = self._infer(scaled_img)
        fx = np.float32(0.8 * w)
        poses_3d, poses_2d = parse_poses(
            inference_result,
            input_scale,
            self.stride,
            fx,
            is_video=False,
        )

        poses = []
        for pose_id in range(len(poses_3d)):
            pose_3d = np.asarray(poses_3d[pose_id]).reshape((19, 4))
            if np.all(pose_3d[:, :3] == -1):
                continue
            pose_2d = np.asarray(poses_2d[pose_id])[:-1].reshape((19, 3))

            keypoints = []
            for kpt_id, name in enumerate(self.KEYPOINT_NAMES):
                confidence = float(pose_2d[kpt_id, 2])

                if confidence == -1:
                    keypoints.append(
                        Keypoint3D(
                            name=name,
                            x=0.0,
                            y=0.0,
                            z=0.0,
                            confidence=0.0,
                            coordinates_3d_meters=None,
                        )
                    )
                    continue

                x_norm = float(pose_2d[kpt_id, 0]) / w
                y_norm = float(pose_2d[kpt_id, 1]) / h

                coord_3d = Coordinate3D(
                    x=round(float(pose_3d[kpt_id, 0]) * self.MODEL_UNITS_TO_METERS, 4),
                    y=round(float(pose_3d[kpt_id, 1]) * self.MODEL_UNITS_TO_METERS, 4),
                    z=round(float(pose_3d[kpt_id, 2]) * self.MODEL_UNITS_TO_METERS, 4),
                )

                keypoints.append(
                    Keypoint3D(
                        name=name,
                        x=round(x_norm, 4),
                        y=round(y_norm, 4),
                        z=coord_3d.z,
                        confidence=round(confidence, 4),
                        coordinates_3d_meters=coord_3d,
                    )
                )

            poses.append(PersonPose(person_id=len(poses) + 1, keypoints=keypoints))

        return PoseResponse(
            model_used="Lightweight 3D Pose Estimation",
            persons_detected=len(poses),
            poses=poses,
        )
