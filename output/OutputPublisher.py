# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import math
from typing import List, Union

import ntcore
from config.config import ConfigStore
from vision_types import CameraPoseObservation, FiducialPoseObservation, ObjDetectObservation, TagAngleObservation


class OutputPublisher:
    def send_apriltag_fps(self, config_store: ConfigStore, timestamp: float, fps: int) -> None:
        raise NotImplementedError

    def send_apriltag_observation(
        self,
        config_store: ConfigStore,
        timestamp: float,
        observation: Union[CameraPoseObservation, None],
        tag_angles: List[TagAngleObservation],
    ) -> None:
        raise NotImplementedError

    def send_objdetect_fps(self, config_store: ConfigStore, timestamp: float, fps: int) -> None:
        raise NotImplementedError

    def send_objdetect_observation(
        self, config_store: ConfigStore, timestamp: float, observations: List[ObjDetectObservation]
    ) -> None:
        raise NotImplementedError


class NTOutputPublisher(OutputPublisher):
    _init_complete: bool = False
    _observations_pub: ntcore.DoubleArrayPublisher
    _apriltags_fps_pub: ntcore.IntegerPublisher
    _objdetect_fps_pub: ntcore.IntegerPublisher
    _objdetect_observations_pub: ntcore.DoubleArrayPublisher

    def _check_init(self, config: ConfigStore):
        # Initialize publishers on first call
        if not self._init_complete:
            self._init_complete = True
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                "/" + str(config.local_config.device_id) + "/output"
            )
            self._observations_pub = nt_table.getDoubleArrayTopic("observations").publish(
                ntcore.PubSubOptions(periodic=0.016667, sendAll=True, keepDuplicates=True, disableRemote=True)
            )
            self._apriltags_fps_pub = nt_table.getIntegerTopic("fps_apriltags").publish()
            self._objdetect_fps_pub = nt_table.getIntegerTopic("fps_objdetect").publish()
            self._objdetect_observations_pub = nt_table.getDoubleArrayTopic("objdetect_observations").publish(
                ntcore.PubSubOptions(periodic=0.016667, sendAll=True, keepDuplicates=True, disableRemote=True)
            )

    def send_apriltag_fps(self, config_store: ConfigStore, timestamp: float, fps: int) -> None:
        self._check_init(config_store)
        self._apriltags_fps_pub.set(fps)

    def send_apriltag_observation(
        self,
        config_store: ConfigStore,
        timestamp: float,
        observation: Union[CameraPoseObservation, None],
        tag_angles: List[TagAngleObservation],
    ) -> None:
        self._check_init(config_store)

        # Send data
        observation_data: List[float] = [0]
        if observation != None:
            observation_data[0] = 1
            observation_data.append(observation.error_0)
            observation_data.append(observation.pose_0.translation().X())
            observation_data.append(observation.pose_0.translation().Y())
            observation_data.append(observation.pose_0.translation().Z())
            observation_data.append(observation.pose_0.rotation().getQuaternion().W())
            observation_data.append(observation.pose_0.rotation().getQuaternion().X())
            observation_data.append(observation.pose_0.rotation().getQuaternion().Y())
            observation_data.append(observation.pose_0.rotation().getQuaternion().Z())
            if observation.error_1 != None and observation.pose_1 != None:
                observation_data[0] = 2
                observation_data.append(observation.error_1)
                observation_data.append(observation.pose_1.translation().X())
                observation_data.append(observation.pose_1.translation().Y())
                observation_data.append(observation.pose_1.translation().Z())
                observation_data.append(observation.pose_1.rotation().getQuaternion().W())
                observation_data.append(observation.pose_1.rotation().getQuaternion().X())
                observation_data.append(observation.pose_1.rotation().getQuaternion().Y())
                observation_data.append(observation.pose_1.rotation().getQuaternion().Z())
        for tag_angle_observation in tag_angles:
            observation_data.append(tag_angle_observation.tag_id)
            for angle in tag_angle_observation.corners.ravel():
                observation_data.append(angle)
            observation_data.append(tag_angle_observation.distance)

        
        self._observations_pub.set(observation_data, math.floor(timestamp * 1000000))

    def send_objdetect_fps(self, config_store: ConfigStore, timestamp: float, fps: int) -> None:
        self._check_init(config_store)
        self._objdetect_fps_pub.set(fps)

    def send_objdetect_observation(
        self, config_store: ConfigStore, timestamp: float, observations: List[ObjDetectObservation], pose: Union[dict, None]
    ) -> None:
        self._check_init(config_store)

        observation_data: List[float] = []
        #find the observation with the highest confidence
        max_confidence = -1.0
        max_confidence_observation = None
        for observation in observations:
            if observation.confidence > max_confidence:
                max_confidence = observation.confidence
                max_confidence_observation = observation    
        if max_confidence_observation is None:        
            return
        observation_data.append(max_confidence_observation.obj_class)
        observation_data.append(max_confidence_observation.confidence)
        for angle in max_confidence_observation.corner_angles.ravel():
            observation_data.append(angle)
            
        #print out the type
        print("Pose type:", type(pose))
        #print out all available keys
        if isinstance(pose, dict):
            print("Pose keys:", pose.keys())
        
        observation_data.append(-1)  # Indicate pose follows
        # Pose can be a CameraPoseObservation or a serialized dict produced by worker
        if (pose is None):
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
            observation_data.append(0)
        else:
            if isinstance(pose, dict[str, any]):
                print("Serialized Pose Data:", pose)
                observation_data.append(pose.get("error_0", 0.0))
                p0 = pose.get("pose_0", {})
                t0 = p0.get("t", (0.0, 0.0, 0.0))
                q0 = p0.get("q", (1.0, 0.0, 0.0, 0.0))
                observation_data.extend([t0[0], t0[1], t0[2], q0[0], q0[1], q0[2], q0[3]])
                observation_data.append(pose.get("error_1", 0.0))
                p1 = pose.get("pose_1", None)
                if p1:
                    t1 = p1.get("t", (0.0, 0.0, 0.0))
                    q1 = p1.get("q", (1.0, 0.0, 0.0, 0.0))
                    observation_data.extend([t1[0], t1[1], t1[2], q1[0], q1[1], q1[2], q1[3]])
                else:
                    observation_data.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        self._objdetect_observations_pub.set(observation_data, math.floor(timestamp * 1000000))
