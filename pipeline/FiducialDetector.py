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
        #These adaptive windows may be needed to be changed for whatever family the Tag is
        #I (grant) have found these values, 5, 21, 8, 7, to be a nice center, but further tuning would be appreciated.
        params.adaptiveThreshWinSizeMin = 5 
        params.adaptiveThreshWinSizeMax = 21
        params.adaptiveThreshWinSizeStep = 8
        params.adaptiveThreshConstant = 7
        #Don't touch these.
        params.minMarkerPerimeterRate = 0.04
        params.minDistanceToBorder = 3 #Do NOT go lower, going this low allows quite a bit of distortion on 92deg lens. 
        
        params.useAruco3Detection = True 
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        params.cornerRefinementMaxIterations = 25  # Good balance for 1.0 scale, more = will work at higher distance, but less FPS
        
        self._aruco_params = params
        
        # ~2.25x speedup
        self._detection_scale = 1.0  # 1600x1304 goes to 1600x1304 right now, but choose as you see fit for more cams. 2 was entirely unneeded any changes.
        

    def detect_fiducials(self, image: cv2.Mat, config_store: ConfigStore) -> List[FiducialImageObservation]:
        if self._detection_scale != 1.0:
            h, w = image.shape[:2]
            scaled_h, scaled_w = int(h * self._detection_scale), int(w * self._detection_scale)
            scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            corners, ids, rejected_corners = cv2.aruco.detectMarkers(scaled_image, self._aruco_dict, parameters=self._aruco_params)
        else:
            corners, ids, rejected_corners = cv2.aruco.detectMarkers(image, self._aruco_dict, parameters=self._aruco_params)
        #refine
        if len(corners) == 0:
            return []
        
        # Scale corners back to original image coordinates, very important for PNP since those intrinsics only work at 1600x1304 / tuned camera resolution
        scale_factor = 1.0 / self._detection_scale
        scaled_corners = [corner * scale_factor for corner in corners]
        
        return [FiducialImageObservation(id[0], corner) for id, corner in zip(ids, scaled_corners)]