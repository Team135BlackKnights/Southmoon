# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from typing import List

import cv2
from config.config import ConfigStore
from vision_types import FiducialImageObservation


class FiducialDetector:
    def __init__(self) -> None:
        raise NotImplementedError

    def detect_fiducials(self, image: cv2.Mat, config_store: ConfigStore) -> List[FiducialImageObservation]:
        raise NotImplementedError


class ArucoFiducialDetector(FiducialDetector):
    def __init__(self, dictionary_id) -> None:
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        params = cv2.aruco.DetectorParameters()
        
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 21
        params.adaptiveThreshWinSizeStep = 8
        params.adaptiveThreshConstant = 7

        params.minMarkerPerimeterRate = 0.04
        params.minDistanceToBorder = 3
        
        params.useAruco3Detection = True
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        params.cornerRefinementMaxIterations = 25  # Good balance for 0.75 scale
        
        self._aruco_params = params
        
        # ~2.25x speedup
        self._detection_scale = 0.75  # 1600x1304 goes to 1200x978
        

    def detect_fiducials(self, image: cv2.Mat, config_store: ConfigStore) -> List[FiducialImageObservation]:
        h, w = image.shape[:2]
        scaled_h, scaled_w = int(h * self._detection_scale), int(w * self._detection_scale)
        scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        corners, ids, rejected_corners = cv2.aruco.detectMarkers(scaled_image, self._aruco_dict, parameters=self._aruco_params)
        #refine
        if len(corners) == 0:
            return []
        
        # Scale corners back to original image coordinates, very important for PNP
        scale_factor = 1.0 / self._detection_scale
        scaled_corners = [corner * scale_factor for corner in corners]
        
        return [FiducialImageObservation(id[0], corner) for id, corner in zip(ids, scaled_corners)]