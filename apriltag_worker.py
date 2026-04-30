#This runs the Apriltag worker, and is by default opened in memeory on EVERY instance of camera

import queue
from typing import List, Tuple, Union

import cv2
from config.config import ConfigStore
from output.overlay_util import overlay_image_observation
from output.StreamServer import MjpegServer
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.PoseEstimator import SquareTargetPoseEstimator
from pipeline.TagAngleCalculator import CameraMatrixTagAngleCalculator
from vision_types import CameraPoseObservation, FiducialImageObservation, FiducialPoseObservation, TagAngleObservation



def apriltag_worker(
    q_in: queue.Queue[Tuple[float, cv2.Mat, ConfigStore]],
    q_out: queue.Queue[
        Tuple[
            float,
            List[FiducialImageObservation],
            Union[CameraPoseObservation, None],
            List[TagAngleObservation],
            Union[FiducialPoseObservation, None],
        ]
    ],
    server_port: int,
):
    '''
    input: 
        a timestamp 
        a frame
        a config store (remote+local)
    output: 
        timestamp completed
        fidicial tags detected (corner included), could be none
        pose data, including which tags were used, both poses (ambig unsolved), and errors. Cannot be none, since errors must be updated.
        tag angles, so TX/TY pairs. Could be none.
    '''
    fiducial_detector = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11) #aruco 3
    camera_pose_estimator = MultiTargetCameraPoseEstimator() #3D SolvePNP
    tag_angle_calculator = CameraMatrixTagAngleCalculator() #undistorts the points for TxTy.
    stream_server = MjpegServer()
    stream_server.start(server_port)

    while True:
        sample = q_in.get() #Blocking
        timestamp: float = sample[0]
        image: cv2.Mat = sample[1]
        config: ConfigStore = sample[2]

        image_observations = fiducial_detector.detect_fiducials(image, config) #only ever done once per frame
        camera_pose_observation = camera_pose_estimator.solve_camera_pose(
            [x for x in image_observations], config
        )
        tag_angle_observations = [
            tag_angle_calculator.calc_tag_angles(x, config) for x in image_observations
        ]
        tag_angle_observations = [x for x in tag_angle_observations if x != None]
        q_out.put(
            (timestamp, image_observations, camera_pose_observation, tag_angle_observations)
        )
        if stream_server.get_client_count() > 0: #if we are CURRENTLY LOOKING AT THIS CAMERA IN A WEB BROWSER
            image = image.copy()
            [overlay_image_observation(image, x) for x in image_observations] #simple OpenCV boxes
            stream_server.set_frame(image)
