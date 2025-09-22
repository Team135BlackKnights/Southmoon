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
    q_in: queue.Queue[Tuple[float, cv2.Mat, ConfigStore]], # inputs a tuple of the timestamp, image, and config
    q_out: queue.Queue[Tuple[float, List[ObjDetectObservation]]], # outputs a tuple of the timestamp and list of observations
    server_port: int, 
):
    object_detector = CoreMLObjectDetector()  

    # Start the MJPEG server for optional live streaming/debug view
    stream_server = MjpegServer() 
    stream_server.start(server_port)

    while True:
        sample = q_in.get() 
        timestamp: float = sample[0]
        image: cv2.Mat = sample[1]
        config: ConfigStore = sample[2]

        observations = object_detector.detect(image, config)

        q_out.put((timestamp, observations)) #puts the timestamp and observations into the output queue
        if stream_server.get_client_count() > 0:
            image = image.copy()
            [overlay_obj_detect_observation(image, x) for x in observations] #goes through all the detections and overlays them on the camera frame
            stream_server.set_frame(image) #sends the frame to the stream server
