# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import queue
from typing import List, Tuple

import cv2
from config.config import ConfigStore
from output.overlay_util import overlay_obj_detect_observation
from output.StreamServer import MjpegServer
from pipeline.ObjectDetector import CoreMLObjectDetector
from vision_types import ObjDetectObservation


def objdetect_worker(
    q_in: queue.Queue[Tuple[float, cv2.Mat, ConfigStore]],
    q_out: queue.Queue[Tuple[float, List[ObjDetectObservation]]],
    server_port: int,
):
    object_detector = CoreMLObjectDetector()
    stream_server = MjpegServer()
    stream_server.start(server_port)
    last_sent_ts = 0.0

    while True:
        sample = q_in.get()
        timestamp: float = sample[0]
        image: cv2.Mat = sample[1]
        config: ConfigStore = sample[2]

        # Optional input throttle to avoid swamping CoreML, respects max_fps if set
        try:
            max_fps = config.local_config.obj_detect_max_fps
        except Exception:
            max_fps = -1
        if max_fps and max_fps > 0:
            min_dt = 1.0 / float(max_fps)
            if (timestamp - last_sent_ts) < min_dt:
                continue
            last_sent_ts = timestamp

        observations = object_detector.detect(image, config)

        q_out.put((timestamp, observations))
        if stream_server.get_client_count() > 0:
            image = image.copy()
            [overlay_obj_detect_observation(image, x) for x in observations]
            stream_server.set_frame(image)
