# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import math
from typing import List, Optional, Tuple, Union

import coremltools
import cv2
import numpy as np
from PIL import Image

from config.config import ConfigStore
from vision_types import ObjDetectObservation
from coremltools import ComputeUnit  # type: ignore


class ObjectDetector:
    def __init__(self) -> None:
        raise NotImplementedError

    def detect(self, image: cv2.Mat, config: ConfigStore) -> List[ObjDetectObservation]:
        raise NotImplementedError


class CoreMLObjectDetector(ObjectDetector):
    """
    CoreML-based detector optimized for Apple Silicon:
      - Loads model once in constructor
      - Reuses a preallocated 640x640 buffer for letterbox/resizing
      - Caches inverse camera matrix for fast corner angle computation
      - Uses predict_batch() when available (keeps NE hot)
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        compute_units: ComputeUnit = ComputeUnit.ALL,
    ) -> None:
        # load model once
        print(f"[CoreMLObjectDetector] Loading model from {model_path} ...")
        self._model: coremltools.models.MLModel = coremltools.models.MLModel(
            model_path, compute_units=compute_units
        )
        print("[CoreMLObjectDetector] Model loaded")

        # preallocated buffer (letterbox target)
        self.input_size = int(input_size)
        self._buffer = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

        # cache for camera matrix inverse to avoid repeated inversion
        self._cached_camera_matrix = None
        self._cached_invK = None
        self._last_lb = None
    def _ensure_invK(self, K: np.ndarray):
        """
        Cache the inverse of the camera matrix K. If K changes, update cache.
        """
        # use object identity if possible, otherwise compare shape+values small cost
        if self._cached_camera_matrix is None or not np.array_equal(self._cached_camera_matrix, K):
            self._cached_camera_matrix = K.copy()
            self._cached_invK = np.linalg.inv(self._cached_camera_matrix)

    def _letterbox_resize_into_buffer(self, image: np.ndarray):
        """
        True letterbox:
          scale = min(S/w, S/h)
          new_w/h = round(w/h * scale)
          pad on BOTH axes
        Returns (buffer, scale, pad_x, pad_y, new_w, new_h)
        """
        h, w = image.shape[:2]
        S = self.input_size

        # IMPORTANT: clear buffer every frame (use YOLO-ish gray padding)
        self._buffer[:] = 114

        scale = min(S / w, S / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # symmetric padding (top-left used for mapping)
        pad_x = (S - new_w) // 2
        pad_y = (S - new_h) // 2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self._buffer[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        self._last_lb = (scale, pad_x, pad_y, new_w, new_h)
        return self._buffer, scale, pad_x, pad_y, new_w, new_h

    @staticmethod
    def _xywh_center_to_xyxy(cx, cy, w, h):
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

    @staticmethod
    def _xywh_topleft_to_xyxy(x, y, w, h):
        return x, y, x + w, y + h

    @staticmethod
    def _unletterbox_xyxy(x1_l, y1_l, x2_l, y2_l, scale, pad_x, pad_y):
        """
        Convert from LETTERBOX pixel coords (0..S) back to original pixels.
        """
        x1 = (x1_l - pad_x) / scale
        y1 = (y1_l - pad_y) / scale
        x2 = (x2_l - pad_x) / scale
        y2 = (y2_l - pad_y) / scale
        return x1, y1, x2, y2
    def detect(self, image: np.ndarray, config: ConfigStore) -> List[ObjDetectObservation]:
        # channel normalization
        if image.ndim == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        h_orig, w_orig = image.shape[:2]

        # preprocess
        input_image, scale, pad_x, pad_y, new_w, new_h = self._letterbox_resize_into_buffer(image)

        pil = Image.fromarray(input_image)

        try:
            predict_batch = getattr(self._model, "predict_batch", None)
            if callable(predict_batch):
                prediction = self._model.predict_batch([{"image": pil}])[0]
            else:
                prediction = self._model.predict({"image": pil})
        except Exception:
            prediction = self._model.predict({"image": pil})

        observations: List[ObjDetectObservation] = []
        if not prediction:
            return observations

        coords = prediction.get("coordinates", None)
        confidences = prediction.get("confidence", None)

        end2end_output = False
        end2end_data = None
        if coords is None or confidences is None:
            if isinstance(prediction, dict) and len(prediction) == 1:
                end2end_data = next(iter(prediction.values()))
                end2end_output = True
            else:
                return observations

        # camera geometry caches
        K = np.array(config.local_config.camera_matrix, dtype=np.float64)
        self._ensure_invK(K)
        invK = self._cached_invK

        min_conf = float(getattr(config.local_config, "obj_detect_min_conf", 0.05))

        def corners_to_angles_and_obs(obj_class: int, confidence: float, x1: float, y1: float, x2: float, y2: float):
            corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)

            corners_in = corners.reshape(-1, 1, 2).astype(np.float64)
            corners_undistorted = cv2.undistortPoints(
                corners_in,
                K,
                config.local_config.distortion_coefficients,
                None,
                K,
            )
            corners_uv = corners_undistorted.reshape(-1, 2)

            ones = np.ones((corners_uv.shape[0], 1), dtype=np.float64)
            homog = np.hstack((corners_uv, ones))
            vecs = (invK @ homog.T).T
            corner_angles = np.arctan(vecs[:, :2])

            observations.append(ObjDetectObservation(obj_class, confidence, corner_angles, corners))

        # ----------------------------
        # NON-END2END: coordinates/confidence heads
        # ----------------------------
        if not end2end_output:
            if coords is None or len(coords) == 0:
                return observations

            for coordinates, confidence_arr in zip(coords, confidences):
                if isinstance(confidence_arr, (list, tuple, np.ndarray)):
                    obj_class = int(np.argmax(confidence_arr))
                    confidence = float(confidence_arr[obj_class])
                else:
                    obj_class = 0
                    confidence = float(confidence_arr)

                if confidence < min_conf:
                    continue

                c = np.asarray(coordinates, dtype=np.float32)
                # Most YOLO-style exports: normalized 0..1 relative to S
                if float(np.max(c)) <= 1.5:
                    cx_l, cy_l, w_l, h_l = (c * self.input_size).tolist()
                else:
                    cx_l, cy_l, w_l, h_l = c.tolist()

                x1_l, y1_l, x2_l, y2_l = self._xywh_center_to_xyxy(cx_l, cy_l, w_l, h_l)
                x1, y1, x2, y2 = self._unletterbox_xyxy(x1_l, y1_l, x2_l, y2_l, scale, pad_x, pad_y)

                # (optional) clamp to image bounds
                x1 = max(0.0, min(x1, w_orig - 1.0))
                x2 = max(0.0, min(x2, w_orig - 1.0))
                y1 = max(0.0, min(y1, h_orig - 1.0))
                y2 = max(0.0, min(y2, h_orig - 1.0))
                if x2 <= x1 or y2 <= y1:
                    continue

                corners_to_angles_and_obs(obj_class, confidence, x1, y1, x2, y2)

            return observations

        # ----------------------------
        # END2END: [N,6] (x1,y1,x2,y2,conf,cls) OR variants
        # ----------------------------
        data = np.asarray(end2end_data)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2 or data.shape[1] < 6:
            return observations

        coords_raw = data[:, :4]
        confs = data[:, 4]
        clses = data[:, 5]

        max_coord = float(np.max(coords_raw)) if coords_raw.size else 0.0
        is_normalized = max_coord <= 1.5

        # If coords are already in original pixel space (rare), detect it:
        # (keeps your old “sometimes it’s already correct” safety net)
        already_orig = (max_coord > self.input_size + 5) and (max_coord <= max(w_orig, h_orig) + 5)

        def to_xyxy(c4, fmt):
            if fmt == "xyxy":
                return float(c4[0]), float(c4[1]), float(c4[2]), float(c4[3])
            if fmt == "xywh_center":
                return self._xywh_center_to_xyxy(float(c4[0]), float(c4[1]), float(c4[2]), float(c4[3]))
            # xywh_top_left
            return self._xywh_topleft_to_xyxy(float(c4[0]), float(c4[1]), float(c4[2]), float(c4[3]))

        def map_to_orig_xyxy(x1_l, y1_l, x2_l, y2_l, mapping):
            if already_orig:
                return x1_l, y1_l, x2_l, y2_l

            if mapping == "direct":
                # no letterbox correction
                x1 = x1_l / self.input_size * w_orig
                x2 = x2_l / self.input_size * w_orig
                y1 = y1_l / self.input_size * h_orig
                y2 = y2_l / self.input_size * h_orig
                return x1, y1, x2, y2

            # correct letterbox unscale/unpad (THIS is what you were missing)
            return self._unletterbox_xyxy(x1_l, y1_l, x2_l, y2_l, scale, pad_x, pad_y)

        fmts = ["xyxy", "xywh_center", "xywh_top_left"]
        mappings = ["letterbox", "direct"]

        # pick the combo that yields the most “valid-looking” boxes
        best_fmt = "xyxy"
        best_mapping = "letterbox"
        best_score = -1

        for fmt in fmts:
            for mapping in mappings:
                valid = 0
                for c4, conf in zip(coords_raw, confs):
                    if float(conf) < min_conf:
                        continue

                    c = (c4 * self.input_size) if is_normalized else c4
                    x1_l, y1_l, x2_l, y2_l = to_xyxy(c, fmt)
                    x1, y1, x2, y2 = map_to_orig_xyxy(x1_l, y1_l, x2_l, y2_l, mapping)

                    if x2 <= x1 or y2 <= y1:
                        continue
                    if x2 < -5 or x1 > w_orig + 5 or y2 < -5 or y1 > h_orig + 5:
                        continue
                    valid += 1

                if valid > best_score:
                    best_score = valid
                    best_fmt = fmt
                    best_mapping = mapping

        for c4, conf, cls in zip(coords_raw, confs, clses):
            confidence = float(conf)
            if confidence < min_conf:
                continue
            obj_class = int(cls)

            c = (c4 * self.input_size) if is_normalized else c4
            x1_l, y1_l, x2_l, y2_l = to_xyxy(c, best_fmt)
            x1, y1, x2, y2 = map_to_orig_xyxy(x1_l, y1_l, x2_l, y2_l, best_mapping)

            # clamp
            x1 = max(0.0, min(x1, w_orig - 1.0))
            x2 = max(0.0, min(x2, w_orig - 1.0))
            y1 = max(0.0, min(y1, h_orig - 1.0))
            y2 = max(0.0, min(y2, h_orig - 1.0))
            if x2 <= x1 or y2 <= y1:
                continue

            corners_to_angles_and_obs(obj_class, confidence, x1, y1, x2, y2)

        return observations



def _quat_to_rotmat(q):
    qw, qx, qy, qz = q
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        raise ValueError("zero quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
    return R


def compute_tx_ty_distance(
    observation: ObjDetectObservation, config: ConfigStore
) -> Optional[Tuple[float, float, float]]:
    """
    Compute tx/ty (degrees) and distance (meters).
    Distance uses size-based estimate from bbox height when possible.
    Positive Ty is up. Returns distance -1.0 if invalid.
    """
    if observation.corner_angles is None:
        return None
    angles = np.asarray(observation.corner_angles, dtype=np.float64)
    if angles.size == 0 or angles.shape[-1] != 2:
        return None

    center = angles.mean(axis=0)
    tx_rad = float(center[0])
    ty_rad = float(center[1])  # OpenCV camera frame: +down

    tx_deg = math.degrees(tx_rad)
    ty_deg = -math.degrees(ty_rad)

    # Size-based distance (prefer for reliability when looking upward)
    if observation.corner_pixels is not None:
        corners_px = np.asarray(observation.corner_pixels, dtype=np.float64)
        if corners_px.size != 0 and corners_px.shape[-1] == 2:
            min_xy = corners_px.min(axis=0)
            max_xy = corners_px.max(axis=0)
            bbox_h = float(max_xy[1] - min_xy[1])
            if bbox_h > 1e-6:
                K = config.local_config.camera_matrix
                if K is not None:
                    K = np.asarray(K, dtype=np.float64)
                    if K.shape == (3, 3):
                        fy = float(K[1, 1])
                        # Use target height (6 in) via 2 * center height config
                        target_z = float(getattr(config.local_config, "obj_detect_target_z_m", 0.0762))
                        target_height = 2.0 * target_z
                        distance_size = (target_height * fy) / bbox_h
                        if distance_size > 0 and math.isfinite(distance_size):
                            return tx_deg, ty_deg, float(distance_size)

    pose = config.remote_config.field_camera_pose
    if pose is None or len(pose) != 7:
        return tx_deg, ty_deg, -1.0

    cam_pos = np.array([pose[0], pose[1], pose[2]], dtype=float)
    cam_quat = (pose[3], pose[4], pose[5], pose[6])

    # Ray in OpenCV camera frame (x right, y down, z forward)
    dir_cv = np.array([math.tan(tx_rad), math.tan(ty_rad), 1.0], dtype=float)
    norm = np.linalg.norm(dir_cv)
    if norm == 0:
        return tx_deg, ty_deg, -1.0
    dir_cv /= norm

    # Convert to WPILib camera frame
    CV_TO_WPI = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    dir_wpi = CV_TO_WPI @ dir_cv

    # Rotate into field frame
    try:
        R_camera_field = _quat_to_rotmat(cam_quat)
    except Exception:
        return tx_deg, ty_deg, -1.0
    dir_field = R_camera_field @ dir_wpi

    dz = float(dir_field[2])
    if abs(dz) < 1e-6:
        return tx_deg, ty_deg, -1.0

    target_z = float(getattr(config.local_config, "obj_detect_target_z_m", 0.0762))
    t = (target_z - float(cam_pos[2])) / dz
    if t <= 0.0 or math.isnan(t) or math.isinf(t):
        return tx_deg, ty_deg, -1.0

    return tx_deg, ty_deg, float(t)
