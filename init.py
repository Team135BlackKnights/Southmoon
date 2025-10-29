# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import argparse
import atexit
import queue
import sys
import threading
import time
from typing import List, Tuple, Union

import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import ntcore

from apriltag_worker import apriltag_worker as apriltag_worker_process_entry
from calibration.CalibrationCommandSource import CalibrationCommandSource, NTCalibrationCommandSource
from calibration.CalibrationSession import CalibrationSession
from config.config import ConfigStore, LocalConfig, RemoteConfig
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from objdetect_worker import objdetect_worker as objdetect_worker_process_entry
from output.OutputPublisher import NTOutputPublisher, OutputPublisher
from output.StreamServer import MjpegServer, StreamServer
from output.overlay_util import *
from output.VideoWriter import FFmpegVideoWriter, VideoWriter
from pipeline.Capture import CAPTURE_IMPLS
import builtins

# (Optional) toggle UMat usage for Metal if available
USE_UMAT = True

class NTLogger:
    def __init__(self, config_store=None):
        if config_store is not None:
            nt_path = "/" + str(config_store.local_config.device_id) + "/config/print_log"
        else:
            nt_path = "/unknown_device/config/print_log"
        self.nt_table = ntcore.NetworkTableInstance.getDefault().getTable(nt_path)
        self.log_key = "log"
        self._buffer = ""

    def write(self, msg):
        self._buffer += str(msg)
        if "\n" in msg:
            try:
                self.nt_table.putString(self.log_key, self._buffer)
            except Exception:
                pass
            self._buffer = ""

    def flush(self):
        if self._buffer:
            try:
                self.nt_table.putString(self.log_key, self._buffer)
            except Exception:
                pass
            self._buffer = ""

def camera_capture_worker(
    capture,
    config_store: ConfigStore,
    remote_config_source: ConfigSource,
    q_out: queue.Queue,  # small control queue carrying (timestamp, success)
    shm_targets: dict,   # dict of {name: (shm_obj, np.ndarray view)}
):
    """
    Dedicated camera capture thread that reads frames and writes into any
    provided shared memory buffers (objdetect / apriltag) and notifies main loop
    via small control queue. Avoids copying frames through Python queues.
    """
    consecutive_failures = 0
    last_config_update = 0
    config_update_interval = 0.1  # Update config every 100ms
    frames_dropped = 0

    while True:
        # Update config periodically (reduces NT load)
        current_time = time.time()
        if current_time - last_config_update > config_update_interval:
            try:
                remote_config_source.update(config_store)
            except Exception:
                pass
            last_config_update = current_time

        timestamp = time.time()
        success, frame = capture.get_frame(config_store)

        if not success:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("Camera capture: Too many consecutive failures, signalling restart.")
                q_out.put((timestamp, False))
                return
            time.sleep(0.2)
            continue

        consecutive_failures = 0

        # If frame shape doesn't match shared memory target shape, skip writing that target
        h, w = frame.shape[:2]

        # Write into each shared memory target's buffer if size matches
        for name, (shm_obj, buf_view) in shm_targets.items():
            try:
                if buf_view.shape[0] == h and buf_view.shape[1] == w and buf_view.shape[2] == frame.shape[2]:
                    # Use np.copyto to avoid creating extra arrays
                    np.copyto(buf_view, frame)
                else:
                    # shapes don't match; skip target
                    pass
            except Exception as e:
                # don't crash capture thread for a worker failure
                if (frames_dropped % 100) == 0:
                    print(f"[camera_capture_worker] write to shm '{name}' failed: {e}")
                frames_dropped += 1
                continue

        # Notify main loop a new frame is available
        try:
            q_out.put_nowait((timestamp, True))
        except queue.Full:
            # main loop is behind — drop notifying but keep capturing (prevents blocking)
            frames_dropped += 1
            if frames_dropped % 100 == 0:
                print(f"[camera_capture_worker] dropped {frames_dropped} frame notifications (main loop slow)")

# Save the original print function
if __name__ == "__main__":
    # Ensure spawn start method on macOS
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        # already set
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--calibration", default="calibration.json")
    args = parser.parse_args()

    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource(args.config, args.calibration)
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()
    local_config_source.update(config)

    original_print = builtins.print
    nt_logger = NTLogger(config_store=config)

    def nt_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        nt_logger.write(msg + "\n")
        original_print(*args, **kwargs)

    builtins.print = nt_print

    capture = CAPTURE_IMPLS[config.local_config.capture_impl]()
    output_publisher: OutputPublisher = NTOutputPublisher()
    video_writer: VideoWriter = FFmpegVideoWriter()
    calibration_session = CalibrationSession()
    calibration_session_server: Union[StreamServer, None] = None

    # Shared memory objects (initialized lazily)
    apriltag_shm = None
    apriltag_buf = None
    objdetect_shm = None
    objdetect_buf = None

    # Worker processes / queues
    apriltag_proc = None
    apriltag_in = None
    apriltag_out = None

    objdetect_proc = None
    objdetect_in = None
    objdetect_out = None

    # Camera notification queue (tiny, only telling main "new frame" and success)
    camera_notify_q = queue.Queue(maxsize=2)

    # Start capture thread (it writes into shared memory targets we create later)
    shm_targets = {}  # name -> (SharedMemory obj, numpy view)
    camera_thread = threading.Thread(
        target=camera_capture_worker,
        args=(capture, config, remote_config_source, camera_notify_q, shm_targets),
        daemon=True,
    )
    camera_thread.start()

    # NetworkTables client init
    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(str(config.local_config.device_id))

    # state & counters
    apriltags_frame_count = 0
    apriltags_last_print = 0
    objdetect_next_frame = -1
    objdetect_frame_count = 0
    objdetect_last_print = 0
    was_calibrating = False
    was_recording = False
    hasStartedObjDetect = False
    hasStartedApriltags = False
    last_image_observations: List = []
    last_objdetect_observations: List = []
    video_frame_cache: List = []

    last_main_config_update = time.time()
    main_config_update_interval = 0.1

    def _cleanup_all():
        # Terminate apriltag process & clean shared memory
        global apriltag_proc, apriltag_shm, apriltag_in, apriltag_out
        global objdetect_proc, objdetect_shm, objdetect_in, objdetect_out

        try:
            if apriltag_in is not None:
                try:
                    apriltag_in.close()
                except Exception:
                    pass
            if apriltag_out is not None:
                try:
                    apriltag_out.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if apriltag_proc is not None and apriltag_proc.is_alive():
                apriltag_proc.terminate()
                apriltag_proc.join(timeout=2)
        except Exception:
            pass
        try:
            if apriltag_shm is not None:
                apriltag_shm.close()
                apriltag_shm.unlink()
        except Exception:
            pass

        try:
            if objdetect_in is not None:
                try:
                    objdetect_in.close()
                except Exception:
                    pass
            if objdetect_out is not None:
                try:
                    objdetect_out.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if objdetect_proc is not None and objdetect_proc.is_alive():
                objdetect_proc.terminate()
                objdetect_proc.join(timeout=2)
        except Exception:
            pass
        try:
            if objdetect_shm is not None:
                objdetect_shm.close()
                objdetect_shm.unlink()
        except Exception:
            pass

    atexit.register(_cleanup_all)

    # Main loop
    while True:
        # Wait for camera thread to signal a new frame (non-blocking-ish)
        try:
            ts, success = camera_notify_q.get(timeout=1.0)
        except queue.Empty:
            print("No frame received from camera thread")
            continue

        # Update config periodically (cheap)
        if time.time() - last_main_config_update > main_config_update_interval:
            try:
                remote_config_source.update(config)
            except Exception:
                pass
            last_main_config_update = time.time()

        # Initialize apriltag shared memory & process lazily
        if config.local_config.apriltags_enable and not hasStartedApriltags:
            h, w = config.remote_config.camera_resolution_height, config.remote_config.camera_resolution_width
            if h > 0 and w > 0:
                try:
                    size = h * w * 3
                    apriltag_shm = shared_memory.SharedMemory(create=True, size=size)
                    apriltag_buf = np.ndarray((h, w, 3), dtype=np.uint8, buffer=apriltag_shm.buf)
                    # create small mp queues for control
                    apriltag_in = mp.Queue(maxsize=2)
                    apriltag_out = mp.Queue(maxsize=2)
                    apriltag_proc = mp.Process(
                        target=apriltag_worker_process_entry,
                        args=(apriltag_shm.name, h, w, apriltag_in, apriltag_out, config.local_config.apriltags_stream_port),
                        daemon=True,
                    )
                    apriltag_proc.start()
                    shm_targets["apriltag"] = (apriltag_shm, apriltag_buf)
                    hasStartedApriltags = True
                    print("[init] Started apriltag worker process")
                except Exception as e:
                    print("[init] Failed to start apriltag worker:", e)

        # Initialize objdetect shared memory & process lazily
        if config.local_config.objdetect_enable and not hasStartedObjDetect:
            h, w = config.remote_config.camera_resolution_height, config.remote_config.camera_resolution_width
            if h > 0 and w > 0:
                try:
                    size = h * w * 3
                    objdetect_shm = shared_memory.SharedMemory(create=True, size=size)
                    objdetect_buf = np.ndarray((h, w, 3), dtype=np.uint8, buffer=objdetect_shm.buf)
                    objdetect_in = mp.Queue(maxsize=2)
                    objdetect_out = mp.Queue(maxsize=2)
                    objdetect_proc = mp.Process(
                        target=objdetect_worker_process_entry,
                        args=(objdetect_shm.name, h, w, objdetect_in, objdetect_out, config.local_config.objdetect_stream_port),
                        daemon=True,
                    )
                    objdetect_proc.start()
                    shm_targets["objdetect"] = (objdetect_shm, objdetect_buf)
                    hasStartedObjDetect = True
                    print("[init] Started objdetect worker process")
                except Exception as e:
                    print("[init] Failed to start objdetect worker:", e)

        if not success:
            print("Camera thread reported failure, restarting process...")
            sys.exit(1)

        # For shared-memory worker model, the camera thread already copied the frame into shared memory.
        # Here we only send a small control message telling worker(s) to run on latest frame.
        if config.local_config.apriltags_enable and hasStartedApriltags:
            try:
                if not apriltag_in.full():
                    apriltag_in.put_nowait((ts, config))
            except Exception:
                pass

            # Try to read back result
            try:
                ts_out, image_observations, pose_observation, tag_angle_observations, demo_pose = apriltag_out.get_nowait()
            except Exception:
                # no result available
                pass
            else:
                output_publisher.send_apriltag_observation(config, ts_out, pose_observation, tag_angle_observations, demo_pose)
                last_image_observations = image_observations

                apriltags_frame_count += 1
                if time.time() - apriltags_last_print > 1.0:
                    output_publisher.send_apriltag_fps(config, ts_out, apriltags_frame_count)
                    apriltags_frame_count = 0
                    apriltags_last_print = time.time()

        # Object detection pipeline
        if config.local_config.objdetect_enable and hasStartedObjDetect:
            try:
                if not objdetect_in.full():
                    objdetect_in.put_nowait((ts, config))
            except Exception:
                pass

            try:
                ts_out, observations, pose = objdetect_out.get_nowait()
            except Exception:
                pass
            else:
                output_publisher.send_objdetect_observation(config, ts_out, observations, pose)
                last_objdetect_observations = observations

                objdetect_frame_count += 1
                dt = time.time() - objdetect_last_print
                if dt >= 1.0:
                    fps = int(round(objdetect_frame_count / dt))
                    output_publisher.send_objdetect_fps(config, ts_out, fps)
                    objdetect_frame_count = 0
                    objdetect_last_print = time.time()

        # Recording + overlays on main process (read latest frame from one of the shm buffers)
        if config.remote_config.is_recording:
            # choose source buffer if available
            if "objdetect" in shm_targets:
                buf = shm_targets["objdetect"][1]
            elif "apriltag" in shm_targets:
                buf = shm_targets["apriltag"][1]
            else:
                # no shm, fall back to capture.get_frame (expensive) - try to avoid this path
                success2, img_for_record = capture.get_frame(config)
                if not success2:
                    img_for_record = None
                else:
                    img_for_record = img_for_record
                buf = None

            if buf is not None:
                # make a copy for video writer to avoid concurrency with capture thread
                img_copy = buf.copy()
            else:
                img_copy = img_for_record

            if img_copy is not None:
                if len(video_frame_cache) >= 2:
                    video_writer.write_frame(ts, video_frame_cache.pop(0), last_image_observations, last_objdetect_observations)
                video_frame_cache.append(img_copy)
        else:
            video_frame_cache = []

        # Calibration handling (unchanged)
        if calibration_command_source.get_calibrating(config):
            if not was_calibrating:
                calibration_session_server = MjpegServer()
                calibration_session_server.start(7999)
            was_calibrating = True
            # prefer to read calibration frame from apriltag shm if present
            if "apriltag" in shm_targets:
                calibration_session.process_frame(shm_targets["apriltag"][1], calibration_command_source.get_capture_flag(config))
                calibration_session_server.set_frame(shm_targets["apriltag"][1])
            else:
                # fallback (rare) - capture a fresh frame
                ok, f = capture.get_frame(config)
                if ok:
                    calibration_session.process_frame(f, calibration_command_source.get_capture_flag(config))
                    calibration_session_server.set_frame(f)
        elif was_calibrating:
            calibration_session.finish()
            sys.exit(0)
