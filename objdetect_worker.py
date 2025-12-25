# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import queue
import traceback
import numpy as np
from multiprocessing import shared_memory
from typing import List, Tuple, Union

import cv2
from config.config import ConfigStore
from output.overlay_util import overlay_obj_detect_observation
from output.StreamServer import MjpegServer
from pipeline.BlenderPoseEstimator import BlenderPoseEstimator
from pipeline.CameraPoseEstimator import MultiBumperCameraPoseEstimator
from pipeline.ObjectDetector import CoreMLObjectDetector
from vision_types import ObjDetectObservation
from vision_types import CameraPoseObservation as CameraPoseObservationType


def objdetect_worker(
    shm_name: str,
    height: int,
    width: int,
    q_in: queue.Queue[tuple[float, ConfigStore]],
    q_out: queue.Queue[tuple[float, List[ObjDetectObservation], dict, str]],
    server_port: int,
):
    """
    Shared-memory object detection worker.
    Receives frames via SharedMemory and lightweight config objects via q_in.

    Workflow:
      - Initialize model and pose estimator on first frame
      - Copy frame data from shared memory (avoid pickling)
      - Run CoreML inference and (pose solve or Blender lookup)
      - Serialize pose into dict for IPC
      - Send results back via q_out
      - Overlay detections for MJPEG server if clients are connected
      - Send observation + serialized pose back via q_out
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buf = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)

    detector = None
    pose_estimator: MultiBumperCameraPoseEstimator | BlenderPoseEstimator | None = None
    stream_server = MjpegServer()
    stream_server.start(server_port)
    last_sent_ts = 0.0
    def _serialize_pose(pose: Union[CameraPoseObservationType, None], debug: str) -> Tuple[dict, str]:
        """Convert pose object into a serializable dict for IPC."""
        if pose is None:
            debug += "\nPose is None; returning None."
            return None, debug
        try:
            #pose is a CameraPoseObservation, cast it
            
            p0_t = pose.pose_0.translation()
            p0_q = pose.pose_0.rotation().getQuaternion()
            out = {
                "tag_ids": pose.tag_ids,
                "error_0": pose.error_0,
                "pose_0": {
                    "t": (p0_t.X(), p0_t.Y(), p0_t.Z()),
                    "q": (p0_q.W(), p0_q.X(), p0_q.Y(), p0_q.Z()),
                },
                "error_1": None,
                "pose_1": None,
            }
            if pose.pose_1 is not None:
                p1_t = pose.pose_1.translation()
                p1_q = pose.pose_1.rotation().getQuaternion()
                out["error_1"] = pose.error_1
                out["pose_1"] = {
                    "t": (p1_t.X(), p1_t.Y(), p1_t.Z()),
                    "q": (p1_q.W(), p1_q.X(), p1_q.Y(), p1_q.Z()),
                }
            return out, debug
        except Exception:
            return None, debug + "\nPose serialization failed."

    while True:
        try:
            # Wait for next job: (timestamp, config)
            timestamp, config = q_in.get()

            if detector is None and config.local_config.obj_detect_model != "":
                model_path = config.local_config.obj_detect_model
                print(f"[ObjDetectWorker] Loading CoreML model: {model_path}")
                detector = CoreMLObjectDetector(model_path)
            if pose_estimator is None:
                print(f"[ObjDetectWorker] Initializing Pose Estimator...")
                if (config.local_config.obj_blender_lookup_csv is not None and
                    config.local_config.obj_blender_lookup_csv != ""):
                    pose_estimator = BlenderPoseEstimator(config.local_config.obj_blender_lookup_csv)
                else:
                    pose_estimator = MultiBumperCameraPoseEstimator()
                    
            max_fps = getattr(config.local_config, "obj_detect_max_fps", -1)
            if max_fps and max_fps > 0:
                min_dt = 1.0 / max_fps
                if (timestamp - last_sent_ts) < min_dt:
                    continue
                last_sent_ts = timestamp

            # Copy frame from shared memory
            image = frame_buf.copy()  
            if detector is None:
                #pose_estimator MUST be BlenderPoseEstimator if no detector
                #use hsv thresholds to find oriented bounding rect
                position,image,debug = pose_estimator.estimate_position(image, config)
                pose_obs,debug = pose_estimator.position_to_field_pose(position,config,debug)
            else:
                observations = detector.detect(image, config)
                if observations is not None and len(observations) >0:  
                    if type(pose_estimator) is BlenderPoseEstimator:
                        #find the biggest CORRECT observation
                        lowest_dist = float('inf')
                        best_position = None
                        for obs in observations:
                            if obs.ai_id == config.remote_config.obj_blender_ai_id:
                                position,debug = pose_estimator.estimate_ai_position(obs, config)
                                xy = np.asarray(position[:2], dtype=float)
                                dist = np.linalg.norm(xy)
                                if dist < lowest_dist:
                                    lowest_dist = dist
                                    best_position = position
                        position = best_position
                        pose_obs,debug = pose_estimator.position_to_field_pose(position,config,debug)
                    else:            
                        pose_obs,debug = pose_estimator.solve_camera_pose(observations, config)
                else:
                    #we have no obs, so pose is None
                    pose_obs = None
                    debug = "NA IO"
            pose_serial,debug = _serialize_pose(pose_obs,debug)
            # Send results to main process
            try:
                q_out.put((timestamp, observations, pose_serial,debug), block=False)
            except queue.Full:
                # Drop oldest if main thread is behind
                try:
                    _ = q_out.get_nowait()
                    q_out.put((timestamp, observations, pose_serial,debug), block=False)
                except Exception:
                    pass

            # MJPEG overlay (only if client connected)
            if stream_server.get_client_count() > 0:
                try:
                    img_disp = image.copy()
                    for obs in observations:
                        overlay_obj_detect_observation(img_disp, obs)
                    stream_server.set_frame(img_disp)
                except Exception as e:
                    print("[ObjDetectWorker] Stream overlay error:", e)
                    traceback.print_exc()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ObjDetectWorker] Error: {e}")
            traceback.print_exc()
            continue

    shm.close()
    stream_server.stop()