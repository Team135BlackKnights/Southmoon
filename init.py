# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.


#imports :)))
import argparse
import queue
import sys
import threading
import time
from typing import List, Tuple, Union

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

#declares parameters that allow for arguments to be passed into the file directly from where the file is run
#config and calibration arguments are created here
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--calibration", default="calibration.json")
    args = parser.parse_args()

    #sets up the limelight config, storing data like pipelines, camera settings, and internal code pathways
    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource(args.config, args.calibration)
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()
    local_config_source.update(config)


    #defines a bunch of key components
    capture = CAPTURE_IMPLS[config.local_config.capture_impl]()    #capture object: returns vision data in a string
    output_publisher: OutputPublisher = NTOutputPublisher()   #the object that allows info to be sent limelight to code and vice versa
    video_writer: VideoWriter = FFmpegVideoWriter()   #video software
    calibration_session = CalibrationSession()    #calibration data
    calibration_session_server: Union[StreamServer, None] = None


    #checks if apriltag detection mode is enabled, sets the queue size for the sub-pub system (if more than one pipeline comes in at once, it'll ignore all but the first)
    if config.local_config.apriltags_enable:
        apriltag_worker_in = queue.Queue(maxsize=1)
        apriltag_worker_out = queue.Queue(maxsize=1)
        #dedicates a thread to running the apriltag_worker function defined in another file
        apriltag_worker = threading.Thread(
            target=apriltag_worker,
            args=(apriltag_worker_in, apriltag_worker_out, config.local_config.apriltags_stream_port),
            daemon=True,
        )
        apriltag_worker.start()

    #same thing down here with ai object detection 
    if config.local_config.objdetect_enable:
        objdetect_worker_in = queue.Queue(maxsize=1)
        objdetect_worker_out = queue.Queue(maxsize=1)
        #dedicates a thread to running the objdetect_worker function defined in another file
        objdetect_worker = threading.Thread(
            target=objdetect_worker,
            args=(objdetect_worker_in, objdetect_worker_out, config.local_config.objdetect_stream_port),
            daemon=True,
        )
        objdetect_worker.start()

    #configures the network table, specifically sets ip and starts software
    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    #convert the ID to string
    ntcore.NetworkTableInstance.getDefault().startClient4(str(config.local_config.device_id))

    #defines a lot of variables that are about to be used in the main logic loop
    #the values associated with these variables aren't important as they are assigned later, only the types are important
    apriltags_frame_count = 0
    apriltags_last_print = 0
    objdetect_next_frame = -1
    objdetect_frame_count = 0
    objdetect_last_print = 0
    was_calibrating = False
    was_recording = False
    last_image_observations: List[FiducialImageObservation] = []
    last_objdetect_observations: List[ObjDetectObservation] = []
    video_frame_cache: List[cv2.Mat] = []

    #starts the main loop in which the limelight does a recording cycle and potentially calibrates
    while True:
        #updates the config, timestamp, and checks if the image was retrieved properly
        remote_config_source.update(config)
        timestamp = time.time()
        success, image = capture.get_frame(config)

        #checks a bunch of bool conditions, if all true then the recording begins
        should_record = (
            success #if the capture object was able to get an image
            and config.remote_config.is_recording #config has recording enabled or not
            and config.remote_config.camera_resolution_width > 0 #camera is actually outputting a 2 dimensional image
            and config.remote_config.camera_resolution_height > 0 
            and config.remote_config.timestamp > 0 #timestamp is updating properly
        )
        #check if it should start recording
        if should_record and not was_recording:
            print("Starting recording")
            video_writer.start(config, len(image.shape) == 2) #checks once again if the camera is recording in two dimensions
        #check if it should stop recording
        elif not should_record and was_recording:
            print("Stopping recording")
            video_writer.stop()
        was_recording = should_record #sets was_recording to the correct state

        #exit if no frame
        if not success:
            print("Found no frame.")
            time.sleep(0.5)
            continue
        
        #checks if calibration mode is enabled in the config
        if calibration_command_source.get_calibrating(config):
            if not was_calibrating: #starts the calibration
                calibration_session_server = MjpegServer() #creates a new calibration server and starts it
                calibration_session_server.start(7999)
            was_calibrating = True #sets was_calibrating to the correct value
            calibration_session.process_frame(image, calibration_command_source.get_capture_flag(config))
            calibration_session_server.set_frame(image)

        #finishes calibration
        elif was_calibrating:
            calibration_session.finish()
            sys.exit(0)
        
        #calibration is not needed, straight to the obj_detect and apriltag pipeline generators
        elif config.local_config.has_calibration:
            if config.local_config.apriltags_enable: #apriltag pipeline enabled
                try:
                    apriltag_worker_in.put((timestamp, image, config), block=False) #publishes frame data to device
                except:  
                    pass #no space in queue
                try:
                    (
                        timestamp_out,
                        image_observations,
                        pose_observation,
                        tag_angle_observations,
                        demo_pose_observation,
                    ) = apriltag_worker_out.get(block=False) 
                except:  
                    pass #no new frames being generated 
                else:
                    #publish observation, a set of frame data that the limelight outputs
                    output_publisher.send_apriltag_observation(
                        config,
                        timestamp_out, 
                        pose_observation, #position
                        tag_angle_observations, #heading
                        demo_pose_observation #¯\_(ツ)_/¯
                    )

                    #most recent observation is saved
                    last_image_observations = image_observations

                    #measures fps
                    fps = None
                    apriltags_frame_count += 1 #increments frame count
                    if time.time() - apriltags_last_print > 1: #keeps the total frame count up to date?
                        apriltags_last_print = time.time()
                        print("Running AprilTag pipeline at", apriltags_frame_count, "fps") #outputs fps
                        output_publisher.send_apriltag_fps(config, timestamp_out, apriltags_frame_count) #publishes fps as extra data
                        apriltags_frame_count = 0

            # Object detection pipeline
            if config.local_config.objdetect_enable:
                # Apply FPS limit for object detection
                if objdetect_next_frame == -1: #tells code at what timestamp the next observation will land
                    objdetect_next_frame = timestamp 
                if config.local_config.obj_detect_max_fps < 0 or timestamp > objdetect_next_frame: #if the current timestamp has surpassed the timestamp at which the previous observation is meant to land, set a new expected timestamp
                    objdetect_next_frame += 1 / config.local_config.obj_detect_max_fps
                    try:
                        objdetect_worker_in.put((timestamp, image, config), block=False)
                    except:  # No space in queue
                        pass
                try:
                    timestamp_out, observations = objdetect_worker_out.get(block=False)
                except:  # No new frames
                    pass
                else:
                    # Publish observation
                    output_publisher.send_objdetect_observation(config, timestamp_out, observations)

                    # Store last observations
                    last_objdetect_observations = observations

                    # Measure FPS
                    fps = None
                    objdetect_frame_count += 1
                    if time.time() - objdetect_last_print > 1:
                        objdetect_last_print = time.time()
                        print("Running object detection pipeline at", objdetect_frame_count, "fps")
                        output_publisher.send_objdetect_fps(config, timestamp, objdetect_frame_count) #publishes fps as extra data
                        objdetect_frame_count = 0

            # Save frame to video
            if config.remote_config.is_recording:
                if len(video_frame_cache) >= 2:
                    # Delay output by two frames to improve alignment with overlays
                    video_writer.write_frame(
                        timestamp, video_frame_cache.pop(0), last_image_observations, last_objdetect_observations
                    )
                video_frame_cache.append(image)
            else:
                video_frame_cache = []

        else:
            # No calibration
            print("No calibration found")
            time.sleep(0.5)
