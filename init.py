#This is the file called by start_cameras.sh, which is run on the Mac recursively if there are any crashes. 
#During development, NTCore will likely fail to be picked up by your IDE. Sadly, you'll need to refer to their docs at
#https://robotpy.readthedocs.io/projects/pyntcore/en/latest/index.html

import argparse
import atexit
import os
import queue
import sys
import threading
import time
from typing import List, Tuple, Union

import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import ntcore
from apriltag_worker import apriltag_worker
from calibration.CalibrationCommandSource import CalibrationCommandSource, NTCalibrationCommandSource
from calibration.CalibrationSession import CalibrationSession
from config.config import ConfigStore, LocalConfig, RemoteConfig
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from objdetect_worker import objdetect_worker
from output.OutputPublisher import NTOutputPublisher, OutputPublisher
from output.StreamServer import MjpegServer, StreamServer
from output.overlay_util import *
from output.VideoWriter import FFmpegVideoWriter, VideoWriter
from pipeline.Capture import CAPTURE_IMPLS
import builtins
#This is a wrapper for the main NT table that the camera publishes to, where every single log to the console is automatically put to 'camera/print_log'
#see nt_print in main for implementation.
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
            self.nt_table.putString(self.log_key, self._buffer)
            self._buffer = ""
    #unused
    def flush(self):
        if self._buffer:
            self.nt_table.putString(self.log_key, self._buffer)
            self._buffer = ""


def camera_capture_worker(
    capture, 
    config_store: ConfigStore, 
    remote_config_source: ConfigSource,
    q_out: queue.Queue[Tuple[float, bool, cv2.Mat]]
):
    """
    Dedicated camera capture thread that continuously reads frames from the camera
    and puts them in a queue. This prevents blocking the main processing loop, but must be done before either OBJ or Apriltag can get updates.
    """
    consecutive_failures = 0
    last_config_update = 0
    config_update_interval = 0.1  # Update config every 100ms instead of every frame
    frames_dropped = 0
    
    while True:
        # Update config from NetworkTables periodically (not every frame to avoid overhead)
        current_time = time.time()
        if current_time - last_config_update > config_update_interval:
            remote_config_source.update(config_store)
            last_config_update = current_time
        
        timestamp = time.time()
        success, image = capture.get_frame(config_store)
        
        if not success:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("Camera capture: Too many consecutive failures, restarting...")
                # Signal the main thread to restart
                q_out.put((timestamp, False, None))
                return
            time.sleep(0.2)
            continue
        
        consecutive_failures = 0
        
        # Put frame in queue (non-blocking). Just skip if queue is full - don't try to drop old frames
        # This is faster and avoids contention with the main loop
        try:
            q_out.put((timestamp, True, image), block=False)
        except queue.Full:
            frames_dropped += 1
            if frames_dropped % 100 == 0:
                print(f"Camera: Dropped {frames_dropped} frames (main loop too slow)")
            pass



#program true start location. Technically, you could call this the Logging and Apriltag Pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json") #adding config as a possible arg
    parser.add_argument("--calibration", default="calibration.json") #adding calib as a possible arg
    args = parser.parse_args() #set them (THIS IS ONE-TIME)
    #for these lines refer to the 'config' folder to see exact procedure.
    config = ConfigStore(LocalConfig(), RemoteConfig()) #set our config to defaults
    local_config_source: ConfigSource = FileConfigSource(args.config, args.calibration) #set our config to the actual configs/calibs
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
    
    #define that we're looking for a USB/Pylon/Gstreamer camera, as well as other output things.
    capture = CAPTURE_IMPLS[config.local_config.capture_impl]()
    output_publisher: OutputPublisher = NTOutputPublisher()
    video_writer: VideoWriter = FFmpegVideoWriter()
    calibration_session = CalibrationSession()
    calibration_session_server: Union[StreamServer, None] = None

    # Camera capture queue and thread - LARGE queue to buffer frames while main loop is busy, for whatever reasons.
    # This causes a "bug" which is handled Robotside, where multiple frames of data can come in at once, causing spikes in processing time. 
    # While that is a problem, it was minor enough for me to ignore it. Wanna improve it?
    camera_queue = queue.Queue(maxsize=30)  # Buffer up to 30 frames (0.5 seconds at 60fps)
    camera_thread = threading.Thread(
        target=camera_capture_worker,
        args=(capture, config, remote_config_source, camera_queue),
        daemon=True,
    )
    #We run the camera in a separate thread for the ability to process older frames while grabbing newer ones. Latency is ~40 ms during comp.
    camera_thread.start()
    #We have now gotten the camera, but we should USE the camera, so we now build the apriltag worker if needed.
    #We do NOT create the object model because things about it can change during runtime, hence it being in the later main loop.
    if config.local_config.apriltags_enable:
        apriltag_worker_in = queue.Queue(maxsize=1)
        apriltag_worker_out = queue.Queue(maxsize=1)
        apriltag_worker = threading.Thread(
            target=apriltag_worker,
            args=(apriltag_worker_in, apriltag_worker_out, config.local_config.apriltags_stream_port),
            daemon=True,
        )
        apriltag_worker.start()

    

    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    #convert the ID to string
    ntcore.NetworkTableInstance.getDefault().startClient4(str(config.local_config.device_id))

    apriltags_frame_count = 0
    apriltags_last_print = 0
    objdetect_next_frame = -1
    objdetect_frame_count = 0
    objdetect_last_print = 0
    was_calibrating = False
    was_recording = False
    hasStartedObjDetect = False
    last_image_observations: List[FiducialImageObservation] = []
    last_objdetect_observations: List[ObjDetectObservation] = []
    video_frame_cache: List[cv2.Mat] = [] #used only for recording.
    
    # Debug timing
    last_debug_print = time.time()
    camera_frame_count = 0
    last_main_config_update = time.time()
    main_config_update_interval = 0.1  # Update config every 100ms in main loop (not every frame!)
    # Objdetect IPC handles (initialized when objdetect starts)
    # This allows Object Detection to run its OWN main thread entirely while communicating with this Main. 
    # This allows for more encapsulation and efficincy, at the cost of fixed memory size. (Hence the queues!) 
    objdetect_worker_in = None
    objdetect_worker_out = None
    objdetect_process = None
    objdetect_shm = None
    def _cleanup_objdetect():
        '''
        DO NOT CALL THIS UNLESS SHUTTING DOWN THE MAIN PROCESS!
        '''
        
        # Close queues
        try:
            if objdetect_worker_in is not None:
                try:
                    objdetect_worker_in.close()
                except Exception:
                    pass
                try:
                    objdetect_worker_in.join_thread()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if objdetect_worker_out is not None:
                try:
                    objdetect_worker_out.close()
                except Exception:
                    pass
                try:
                    objdetect_worker_out.join_thread()
                except Exception:
                    pass
        except Exception:
            pass
        # Terminate process
        try:
            if objdetect_process is not None and objdetect_process.is_alive():
                try:
                    objdetect_process.terminate()
                except Exception:
                    pass
                try:
                    objdetect_process.join(timeout=2)
                except Exception:
                    pass
        except Exception:
            pass
        # Unlink shared memory
        try:
            if objdetect_shm is not None:
                try:
                    objdetect_shm.close()
                except Exception:
                    pass
                try:
                    objdetect_shm.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    atexit.register(_cleanup_objdetect) #on exit for any reason including exceptions, run that function before closing.
    
    while True: #Main thread!
        # Get frame from camera capture thread FIRST (blocking, but camera runs in parallel)
        # Don't do ANY work before getting the frame to maximize consumption rate
        try:
            timestamp, success, image = camera_queue.get(timeout=1.0)
        except queue.Empty:
            print("No frame received from camera thread")
            continue
        
        # Update config less frequently to reduce overhead (do this AFTER getting frame to avoid losing frames)
        if time.time() - last_main_config_update > main_config_update_interval:
            remote_config_source.update(config)
            last_main_config_update = time.time()
        if config.local_config.objdetect_enable and config.remote_config.camera_id != "" and not hasStartedObjDetect:
            h, w = config.remote_config.camera_resolution_height, config.remote_config.camera_resolution_width
            if h > 0 and w > 0:
                # Create shared memory for RGB frames and assign to outer var for cleanup
                objdetect_shm = shared_memory.SharedMemory(create=True, size=h * w * 3)  # 3 channels RGB
                buf = np.ndarray((h, w, 3), dtype=np.uint8, buffer=objdetect_shm.buf)

                objdetect_worker_in = mp.Queue(maxsize=2)
                objdetect_worker_out = mp.Queue(maxsize=2)

                objdetect_process = mp.Process(
                    target=objdetect_worker,
                    args=(objdetect_shm.name, h, w, objdetect_worker_in, objdetect_worker_out, config.local_config.objdetect_stream_port),
                )
                objdetect_process.start()
                hasStartedObjDetect = True
        

        # Handle camera failure
        if not success:
            print("Camera thread reported failure, restarting process...")
            sys.exit(1)

        # Start and stop recording, only during a match / forced as a reminder.
        should_record = (
            success
            and config.remote_config.is_recording
            and config.remote_config.camera_resolution_width > 0
            and config.remote_config.camera_resolution_height > 0
            and config.remote_config.timestamp > 0
        )
        if should_record and not was_recording:
            print("Starting recording")
            video_writer.start(config, len(image.shape) == 2)
        elif not should_record and was_recording:
            print("Stopping recording")
            video_writer.stop()
        was_recording = should_record
        
        
        #Calibration commands.
        if calibration_command_source.get_calibrating(config):
            # Calibration mode
            if not was_calibrating:
                calibration_session_server = MjpegServer()
                calibration_session_server.start(7999)
            was_calibrating = True
            calibration_session.process_frame(image, calibration_command_source.get_capture_flag(config))
            calibration_session_server.set_frame(image)

        elif was_calibrating:
            # Finish calibration
            calibration_session.finish()
            sys.exit(0) #we exit to force the clearing of the file. Refer to the readme for what to do from here if you have just calibrated.

        elif config.local_config.has_calibration:
            # AprilTag pipeline first
            if config.local_config.apriltags_enable:
                try:
                    apriltag_worker_in.put((timestamp, image, config), block=False)
                except:  # No space in queue likely, or some unknown bug, but we NEED to catch it and ignore it to tell the robot "no data" 
                    pass
                try:
                    (
                        timestamp_out,
                        image_observations,
                        pose_observation,
                        tag_angle_observations,
                    ) = apriltag_worker_out.get(block=False)
                except:  # No new frames likely.
                    pass
                else:
                    # Publish observation
                    output_publisher.send_apriltag_observation(
                        config, timestamp_out, pose_observation, tag_angle_observations
                    ) #This encapsulates the output into a bunch of numbers for NT.

                    # Store last observations for recording purposes (showing the tags in the recording is cool!)
                    last_image_observations = image_observations

                    # Measure FPS
                    fps = None
                    apriltags_frame_count += 1
                    if time.time() - apriltags_last_print > 1:
                        apriltags_last_print = time.time()
                        #print("Running AprilTag pipeline at", apriltags_frame_count, "fps")
                        output_publisher.send_apriltag_fps(config, timestamp_out, apriltags_frame_count)
                        apriltags_frame_count = 0

            # Object detection pipeline
            if config.local_config.objdetect_enable and hasStartedObjDetect:
                try:
                    np.copyto(buf, image) #fill the other "main" running the object detection with the image.

                    if objdetect_worker_in.full():
                        _ = objdetect_worker_in.get_nowait() #Ignore a frame on purpose, since we are full.
                    objdetect_worker_in.put_nowait((timestamp, config)) #"Hey Object Pipeline! You have new data present!"
                except Exception as e:
                    print(f"[ObjDetect] Dropped frame: {e}") #Unknown frame drop.
                    pass

                try:
                    timestamp_out, observations, pose, debug, txy_observations = objdetect_worker_out.get_nowait() 
                    #Forces the object pipeline to either have something NOW, or wait until the next entire loop.
                    #THIS is how we get differing FPS for object and apriltag pipelines, since it can force ignore OBJ. 
                except queue.Empty:
                    pass #No data/data still processing
                except Exception as e:
                    print("[WARN] Object detection IPC read failed:", e) #Did we just lose the Object detection pipeline?!?
                else:
                    if not getattr(config.local_config, "objdetect_tx_ty_only", False): #If we HAVE pose data, send it.
                        output_publisher.send_objdetect_observation(config, timestamp_out, observations, pose)
                    output_publisher.send_objdetect_txy(config, timestamp_out, txy_observations)
                    if (debug != ""): #If the OBJ thread has something new to say, say it in this thread to get logged by NT.
                        print("[ObjDetect]:", debug)
                    last_objdetect_observations = observations #for recording object boxes/confidence.

                    #Recording FPS
                    objdetect_frame_count += 1
                    dt = time.time() - objdetect_last_print
                    if dt >= 1.0:
                        fps = objdetect_frame_count / dt
                        # to nearest int
                        fps = int(round(fps))
                        output_publisher.send_objdetect_fps(config, timestamp, fps)
                        objdetect_frame_count = 0
                        objdetect_last_print = time.time()
            # Save most recent frame + observation(s) to video
            if config.remote_config.is_recording:
                if len(video_frame_cache) >= 2:
                    # Delay output by two frames to hopefully improve alignment with overlays, SPECIFICALLY obj. (remember, the whole siystem is around 40ms delay.)
                    # Also just prevents some weird skipping issues.
                    video_writer.write_frame(
                        timestamp, video_frame_cache.pop(0), last_image_observations, last_objdetect_observations
                    ) #a real use of pop!
                video_frame_cache.append(image)
            else:
                video_frame_cache = []

        else:
            # No calibration
            print("No calibration found")
            time.sleep(0.5)
