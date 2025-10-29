# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import queue
from typing import List, Tuple, Union
import traceback

import cv2
from multiprocessing import shared_memory
import numpy as np

from config.config import ConfigStore
from output.overlay_util import overlay_image_observation
from output.StreamServer import MjpegServer
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.PoseEstimator import SquareTargetPoseEstimator
from pipeline.TagAngleCalculator import CameraMatrixTagAngleCalculator
from vision_types import CameraPoseObservation, FiducialImageObservation, FiducialPoseObservation, TagAngleObservation

DEMO_ID = 42

def apriltag_worker(
    shm_name: str,
    height: int,
    width: int,
    q_in: "queue.Queue[tuple[float, ConfigStore]]",
    q_out: "queue.Queue",
    server_port: int,
):
    """
    Process entrypoint for apriltag detection.
    Reads frames from shared memory (no pickling), performs fiducial detection,
    pose solves, and publishes results via q_out. Also serves MJPEG overlays.
    """
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        frame_buf = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
    except Exception as e:
        print("[apriltag_worker] Failed to open shared memory:", e)
        return

    fiducial_detector = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11)
    camera_pose_estimator = MultiTargetCameraPoseEstimator()
    tag_angle_calculator = CameraMatrixTagAngleCalculator()
    tag_pose_estimator = SquareTargetPoseEstimator()
    stream_server = MjpegServer()
    stream_server.start(server_port)

    while True:
        try:
            ts, config = q_in.get()

            # Use view on shared memory; avoid copying unless overlay needed
            try:
                image = frame_buf  # direct view
                # Optionally convert to UMat if available (speeds CV ops on some builds)
                try:
                    if hasattr(cv2, "UMat"):
                        image_umat = cv2.UMat(image)
                        image_for_detect = image_umat
                    else:
                        image_for_detect = image
                except Exception:
                    image_for_detect = image
            except Exception:
                # If shared memory read fails, skip this job
                continue

            # Detect fiducials (implementation likely C/C++ heavy)
            image_observations = fiducial_detector.detect_fiducials(image_for_detect, config)
            camera_pose_observation = camera_pose_estimator.solve_camera_pose(
                [x for x in image_observations if x.tag_id != DEMO_ID], config
            )
            tag_angle_observations = [
                tag_angle_calculator.calc_tag_angles(x, config) for x in image_observations if x.tag_id != DEMO_ID
            ]
            tag_angle_observations = [x for x in tag_angle_observations if x is not None]
            demo_image_observations = [x for x in image_observations if x.tag_id == DEMO_ID]
            demo_pose_observation: Union[FiducialPoseObservation, None] = None
            if len(demo_image_observations) > 0:
                demo_pose_observation = tag_pose_estimator.solve_fiducial_pose(demo_image_observations[0], config)

            # Send observation back to main process (non-blocking)
            try:
                q_out.put_nowait((ts, image_observations, camera_pose_observation, tag_angle_observations, demo_pose_observation))
            except Exception:
                # If out queue is full, drop the oldest and replace it (keep latest)
                try:
                    _ = q_out.get_nowait()
                    q_out.put_nowait((ts, image_observations, camera_pose_observation, tag_angle_observations, demo_pose_observation))
                except Exception:
                    pass

            # If clients are connected, prepare an overlayed image and send to MJPEG server
            try:
                if stream_server.get_client_count() > 0:
                    img_disp = image.copy()
                    [overlay_image_observation(img_disp, x) for x in image_observations]
                    stream_server.set_frame(img_disp)
            except Exception as e:
                print("[apriltag_worker] stream overlay error:", e)
                traceback.print_exc()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[apriltag_worker] Error:", e)
            traceback.print_exc()
            # continue processing subsequent frames
            continue

    try:
        shm.close()
    except Exception:
        pass
    stream_server.stop()
