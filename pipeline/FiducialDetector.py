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
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7

        # Ignore tiny/noisy quads and those too close to borders
        #params.minMarkerPerimeterRate = 0.1   # default ~0.03; increase to ignore small blobs
        #params.maxMarkerPerimeterRate = 1.0
        #params.minGroupDistance = 25
        params.minDistanceToBorder = 5
        params.useAruco3Detection = True
       # params.minCornerDistanceRate = 0.10
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        params.cornerRefinementMaxIterations = 15   
        self._aruco_params = params
        

    def detect_fiducials(self, image: cv2.Mat, config_store: ConfigStore) -> List[FiducialImageObservation]:
        corners, ids, _ = cv2.aruco.detectMarkers(image, self._aruco_dict, parameters=self._aruco_params)
        if len(corners) == 0:
            return []
        return [FiducialImageObservation(id[0], corner) for id, corner in zip(ids, corners)]
