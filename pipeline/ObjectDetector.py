# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import math
from typing import List, Optional, Union

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

    def _ensure_invK(self, K: np.ndarray):
        """
        Cache the inverse of the camera matrix K. If K changes, update cache.
        """
        # use object identity if possible, otherwise compare shape+values small cost
        if self._cached_camera_matrix is None or not np.array_equal(self._cached_camera_matrix, K):
            self._cached_camera_matrix = K.copy()
            self._cached_invK = np.linalg.inv(self._cached_camera_matrix)

    def _letterbox_resize_into_buffer(self, image: np.ndarray) -> np.ndarray:
        """
        Resize (with aspect-preserving letterbox) directly into the preallocated buffer.
        Returns the buffer (same object each call).
        """
        h, w = image.shape[:2]

        # compute target scaled height while keeping width=input_size
        scaled_height = int(self.input_size / (w / h))
        # clamp to [1, input_size]
        scaled_height = max(1, min(self.input_size, scaled_height))
        bar_height = (self.input_size - scaled_height) // 2

        # resize into the buffer's slice to avoid allocating a new array
        # cv2.resize supports dst parameter to write into slice memory
        # ensure correct dtype and contiguous slice
        dst_slice = self._buffer[bar_height : bar_height + scaled_height, 0 : self.input_size]
        # cv2.resize does not accept shape mismatch dst; use work-around: resize to exact shape
        resized = cv2.resize(image, (self.input_size, scaled_height))
        dst_slice[:] = resized  # copy resized into buffer slice

        # if image has alpha or gray, ensure 3 channels already handled upstream
        return self._buffer

    def detect(self, image: np.ndarray, config: ConfigStore) -> List[ObjDetectObservation]:
        # Convert greyscale to RGB if needed
        if image.ndim == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Create letterboxed input (reused buffer to avoid allocations)
        input_image = self._letterbox_resize_into_buffer(image)

        # Run CoreML model - use predict_batch if available to keep compute unit hot
        # CoreML expects a PIL Image for many vision models; creating the PIL
        # object is unavoidable in many cases, but we keep it tight.
        pil = Image.fromarray(input_image)

        prediction = None
        try:
            # prefer predict_batch (returns list of dicts)
            predict_batch = getattr(self._model, "predict_batch", None)
            if callable(predict_batch):
                batch_out = self._model.predict_batch([{"image": pil}])
                # batch_out is a list of outputs; take first
                prediction = batch_out[0]
            else:
                prediction = self._model.predict({"image": pil})
        except Exception as e:
            # fallback to single predict if predict_batch fails for any reason
            prediction = self._model.predict({"image": pil})

        observations: List[ObjDetectObservation] = []

        # If model returns no detections, be robust
        if prediction is None:
            return observations

        coords = prediction.get("coordinates", [])
        confidences = prediction.get("confidence", [])

        if len(coords) == 0:
            return observations

        # prepare camera geometry caches
        K = np.array(config.local_config.camera_matrix, dtype=np.float64)
        self._ensure_invK(K)
        invK = self._cached_invK

        h_orig, w_orig = image.shape[:2]
        scaled_height = int(self.input_size / (w_orig / h_orig))
        bar_height = (self.input_size - scaled_height) // 2
        # scaling factors used to map model coords back to original image
        # model x coords are normalized to width, y coords normalized to input_size (640)
        x_scale = w_orig
        y_scale = scaled_height / self.input_size * h_orig  # careful: mapping via letterbox
        y_offset_pixels = bar_height

        # iterate predictions
        for coordinates, confidence_arr in zip(coords, confidences):
            # coordinates format: [x_center_norm, y_center_norm, width_norm, height_norm]
            # class selection
            if isinstance(confidence_arr, (list, tuple, np.ndarray)):
                obj_class = int(np.argmax(confidence_arr))
                confidence = float(confidence_arr[obj_class])
            else:
                # If model emits single-class score or different format
                obj_class = 0
                confidence = float(confidence_arr)

            # Map normalized model coordinates back to original image pixels
            cx = float(coordinates[0]) * x_scale
            cy = (float(coordinates[1]) * self.input_size - y_offset_pixels) / scaled_height * h_orig
            w_box = float(coordinates[2]) * x_scale
            # Note: model height coordinate may be normalized to input_size; adjust accordingly
            h_box = float(coordinates[3]) / (scaled_height / self.input_size) * h_orig

            # construct corner coordinates in original image pixel space
            x_min = cx - w_box / 2.0
            x_max = cx + w_box / 2.0
            y_min = cy - h_box / 2.0
            y_max = cy + h_box / 2.0

            corners = np.array(
                [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]], dtype=np.float32
            )

            # cv2.undistortPoints expects shape (N,1,2)
            corners_in = corners.reshape(-1, 1, 2).astype(np.float64)
            corners_undistorted = cv2.undistortPoints(
                corners_in,
                K,
                config.local_config.distortion_coefficients,
                None,
                K,
            )  # returns (N,1,2)

            # Flatten to (N,2)
            corners_uv = corners_undistorted.reshape(-1, 2)

            # Build homogeneous coords (N,3) for vectorized invK multiply
            ones = np.ones((corners_uv.shape[0], 1), dtype=np.float64)
            homog = np.hstack((corners_uv, ones))  # shape (4,3)

            # Multiply invK (3x3) by each homogeneous column -> result (4,3)
            vecs = (invK @ homog.T).T  # shape (4,3)

            # Compute corner angles (atan of x,z? The original used atan(vec[0]) and atan(vec[1]))
            # If vec is [X, Y, Z], dividing by Z might be intended, but original code used atan(vec[0]).
            # We'll replicate original behavior: atan(X) and atan(Y)
            corner_angles = np.arctan(vecs[:, :2])  # shape (4,2)

            observations.append(ObjDetectObservation(obj_class, confidence, corner_angles, corners))

        return observations
