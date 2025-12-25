# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from dataclasses import dataclass,field

import numpy
import numpy.typing


@dataclass
class LocalConfig:
    device_id: str = ""
    server_ip: str = ""
    apriltags_stream_port: int = 8000
    objdetect_stream_port: int = 8001
    capture_impl: str = ""
    obj_detect_model: str = ""
    obj_blender_lookup_csv: str = ""
    obj_use_oriented_detection: bool = False    
    obj_detect_max_fps: int = -1
    apriltags_enable: bool = False
    objdetect_enable: bool = True
    video_folder: str = ""
    has_calibration: bool = False
    camera_matrix: numpy.typing.NDArray[numpy.float64] = None
    distortion_coefficients: numpy.typing.NDArray[numpy.float64] = None


@dataclass
class RemoteConfig:
    camera_id: str = ""
    camera_location: str = ""
    camera_resolution_width: int = 1600
    camera_resolution_height: int = 1304
    camera_auto_exposure: int = 0
    camera_auto_white_balance: int = 0
    camera_exposure: float = 0
    camera_saturation: int = 0
    camera_hue: int = 0
    camera_white_balance: int = 0
    camera_gain: float = 0
    fiducial_size_m: float = 0
    tag_layout: any = None
    field_camera_pose: list[float] = field(default_factory=lambda: [0,0,0,0,0,0,0])
    is_recording: bool = False
    timestamp: int = 0
    lower_hsv: list[int] = field(default_factory=lambda: [0,0,0])
    upper_hsv: list[int] = field(default_factory=lambda: [0,0,0])
    obj_blender_ai_id: int = -1


@dataclass
class ConfigStore:
    local_config: LocalConfig
    remote_config: RemoteConfig
