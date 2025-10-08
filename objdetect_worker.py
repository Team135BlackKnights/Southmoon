# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import queue
import traceback
from typing import List, Tuple

import cv2
from config.config import ConfigStore
from output.overlay_util import overlay_obj_detect_observation
from output.StreamServer import MjpegServer
from pipeline.CameraPoseEstimator import MultiBumperCameraPoseEstimator
from pipeline.ObjectDetector import CoreMLObjectDetector
from vision_types import ObjDetectObservation
from pipeline.PoseEstimator import SquareTargetPoseEstimator
from vision_types import FiducialPoseObservation

def objdetect_worker(
    q_in: queue.Queue[Tuple[float, cv2.Mat, ConfigStore]],
    q_out: queue.Queue[Tuple[float, List[ObjDetectObservation]]],
    server_port: int,
):
    """
    Worker loop:
      - constructs a single CoreMLObjectDetector once (model path read from first config)
      - respects optional max_fps throttle (skips frames to keep NE/CPU hot instead of overloading)
      - overlays detections only when there are clients
    """
    # block until first sample to obtain config/model path
    first_sample = q_in.get()
    timestamp_first, image_first, config_first = first_sample

    model_path = config_first.local_config.obj_detect_model
    # If compute_unit is provided and matches CoreML ComputeUnit names, use it;
    # otherwise CoreMLObjectDetector default (CPU_AND_NE) is used.
    detector = CoreMLObjectDetector(model_path)
    bumper_pose_estimator = MultiBumperCameraPoseEstimator()

    stream_server = MjpegServer()
    stream_server.start(server_port)
    last_sent_ts = 0.0

    # process the first sample immediately (we already have it)
    try:
        observations = detector.detect(image_first, config_first)
        q_out.put((timestamp_first, observations))
        if stream_server.get_client_count() > 0:
            img_copy = image_first.copy()
            for obs in observations:
                overlay_obj_detect_observation(img_copy, obs)
            stream_server.set_frame(img_copy)
        last_sent_ts = timestamp_first
    except Exception:
        # if first detection fails, continue to loop normally
        pass

    # main loop - now we can non-blocking wait for frames
    while True:
        sample = q_in.get()  # block until next sample
        timestamp: float = sample[0]
        image: cv2.Mat = sample[1]
        config: ConfigStore = sample[2]

        # Optional input throttle to avoid swamping CoreML; respects max_fps if set
        max_fps = -1.0
        try:
            max_fps = float(config.local_config.obj_detect_max_fps)
        except Exception:
            max_fps = -1.0

        if max_fps and max_fps != 0:
            min_dt = 1.0 / max_fps
            if (timestamp - last_sent_ts) < min_dt:
                # skip frame to respect max_fps
                continue
            last_sent_ts = timestamp

        # Run detection
        observations = detector.detect(image, config)
        pose = bumper_pose_estimator.solve_camera_pose(observations, config)
        # Put results on output queue (non-blocking if consumer is slow)
        try:
            q_out.put((timestamp, observations, pose), block=False)
        except queue.Full:
            # If output queue is full, drop the oldest and push current (keep recent)
            try:
                _ = q_out.get_nowait()
                q_out.put((timestamp, observations, pose), block=False)
            except Exception:
                pass

        # Only overlay and stream if client connected to avoid wasted work
        try:
            if stream_server.get_client_count() > 0:
                img_copy = image.copy()
                for obs in observations:
                    overlay_obj_detect_observation(img_copy, obs)
                stream_server.set_frame(img_copy)
        except Exception as e:
            # Do not crash worker on overlay/stream errors
            print(e)
            traceback.print_exc()

            pass
