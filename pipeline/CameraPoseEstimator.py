# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from typing import List, Union

import cv2
import numpy
from config.config import ConfigStore
from pipeline.coordinate_systems import openCvPoseToWpilib, wpilibTranslationToOpenCv
from vision_types import CameraPoseObservation, FiducialImageObservation, ObjDetectObservation
from wpimath.geometry import *


class CameraPoseEstimator:
    def __init__(self) -> None:
        raise NotImplementedError

    def solve_camera_pose(
        self, image_observations: List[FiducialImageObservation], config_store: ConfigStore
    ) -> Union[CameraPoseObservation, None]:
        raise NotImplementedError

class MultiBumperCameraPoseEstimator(CameraPoseEstimator):
    """
    Estimates camera-to-bumper poses for FRC bumpers using a list of ObjDetectObservation.
    Mimics MultiTargetCameraPoseEstimator but without field layout.
    """

    def __init__(self, bumper_size_m: float = 0.8382):
        self.bumper_size_m = bumper_size_m
        half = bumper_size_m / 2.0
        # Object points in bumper frame
        self.object_points = numpy.array(
            [
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=numpy.float64,
        )

    def solve_camera_pose(
        self, image_observations: List[ObjDetectObservation], config_store: ConfigStore
    ) -> Union[CameraPoseObservation, None]:

        if len(image_observations) == 0:
            return None

        poses = []
        errors = []

        for i, obs in enumerate(image_observations):
            if obs.corner_pixels is None or len(obs.corner_pixels) != 4:
                continue

            # OpenCV expects (N,1,2)
            image_points = numpy.array(obs.corner_pixels, dtype=numpy.float64).reshape(-1, 1, 2)

            try:
                _, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
                    self.object_points,
                    image_points,
                    config_store.local_config.camera_matrix,
                    config_store.local_config.distortion_coefficients,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            except Exception as e:
                print(f"[MultiBumperCameraPoseEstimator] solvePnP failed: {e}")
                continue

            # Convert first solution to Pose3d
            pose_0 = openCvPoseToWpilib(tvecs[0], rvecs[0])
            err_0 = float(reproj_errors[0][0]) if reproj_errors is not None else 0.0

            # Alternate solution (planar ambiguity)
            pose_1, err_1 = None, None
            if len(tvecs) > 1:
                pose_1 = openCvPoseToWpilib(tvecs[1], rvecs[1])
                err_1 = float(reproj_errors[1][0]) if reproj_errors is not None else 0.0

            # Camera relative to bumper center
            poses.append((pose_0, pose_1))
            errors.append((err_0, err_1))

        if len(poses) == 0:
            return None

        # Highest confidence observation
        best_idx = numpy.argmin([e[0] for e in errors])
        pose_0, pose_1 = poses[best_idx]
        err_0, err_1 = errors[best_idx]

        return CameraPoseObservation(
            tag_ids=[0],  # dummy ID since we don't have field IDs
            pose_0=pose_0,
            error_0=err_0,
            pose_1=pose_1,
            error_1=err_1,
        )
class MultiTargetCameraPoseEstimator(CameraPoseEstimator):
    def __init__(self) -> None:
        pass

    def solve_camera_pose(
        self, image_observations: List[FiducialImageObservation], config_store: ConfigStore
    ) -> Union[CameraPoseObservation, None]:
        # Exit if no tag layout available
        if config_store.remote_config.tag_layout == None:
            return None

        # Exit if no observations available
        if len(image_observations) == 0:
            return None

        # Create set of object and image points
        fid_size = config_store.remote_config.fiducial_size_m
        object_points = []
        image_points = []
        tag_ids = []
        tag_poses = []
        for observation in image_observations:
            tag_pose = None
            for tag_data in config_store.remote_config.tag_layout["tags"]:
                if tag_data["ID"] == observation.tag_id:
                    tag_pose = Pose3d(
                        Translation3d(
                            tag_data["pose"]["translation"]["x"],
                            tag_data["pose"]["translation"]["y"],
                            tag_data["pose"]["translation"]["z"],
                        ),
                        Rotation3d(
                            Quaternion(
                                tag_data["pose"]["rotation"]["quaternion"]["W"],
                                tag_data["pose"]["rotation"]["quaternion"]["X"],
                                tag_data["pose"]["rotation"]["quaternion"]["Y"],
                                tag_data["pose"]["rotation"]["quaternion"]["Z"],
                            )
                        ),
                    )
            if tag_pose != None:
                # Add object points by transforming from the tag center
                corner_0 = tag_pose + Transform3d(Translation3d(0, fid_size / 2.0, -fid_size / 2.0), Rotation3d())
                corner_1 = tag_pose + Transform3d(Translation3d(0, -fid_size / 2.0, -fid_size / 2.0), Rotation3d())
                corner_2 = tag_pose + Transform3d(Translation3d(0, -fid_size / 2.0, fid_size / 2.0), Rotation3d())
                corner_3 = tag_pose + Transform3d(Translation3d(0, fid_size / 2.0, fid_size / 2.0), Rotation3d())
                object_points += [
                    wpilibTranslationToOpenCv(corner_0.translation()),
                    wpilibTranslationToOpenCv(corner_1.translation()),
                    wpilibTranslationToOpenCv(corner_2.translation()),
                    wpilibTranslationToOpenCv(corner_3.translation()),
                ]

                # Add image points from observation
                image_points += [
                    [observation.corners[0][0][0], observation.corners[0][0][1]],
                    [observation.corners[0][1][0], observation.corners[0][1][1]],
                    [observation.corners[0][2][0], observation.corners[0][2][1]],
                    [observation.corners[0][3][0], observation.corners[0][3][1]],
                ]

                # Add tag ID and pose
                tag_ids.append(observation.tag_id)
                tag_poses.append(tag_pose)

        # Single tag, return two poses
        if len(tag_ids) == 1:
            object_points = numpy.array(
                [
                    [-fid_size / 2.0, fid_size / 2.0, 0.0],
                    [fid_size / 2.0, fid_size / 2.0, 0.0],
                    [fid_size / 2.0, -fid_size / 2.0, 0.0],
                    [-fid_size / 2.0, -fid_size / 2.0, 0.0],
                ]
            )
            try:
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    object_points,
                    numpy.array(image_points),
                    config_store.local_config.camera_matrix,
                    config_store.local_config.distortion_coefficients,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            except:
                return None

            # Calculate WPILib camera poses
            field_to_tag_pose = tag_poses[0]
            camera_to_tag_pose_0 = openCvPoseToWpilib(tvecs[0], rvecs[0])
            camera_to_tag_pose_1 = openCvPoseToWpilib(tvecs[1], rvecs[1])
            camera_to_tag_0 = Transform3d(camera_to_tag_pose_0.translation(), camera_to_tag_pose_0.rotation())
            camera_to_tag_1 = Transform3d(camera_to_tag_pose_1.translation(), camera_to_tag_pose_1.rotation())
            field_to_camera_0 = field_to_tag_pose.transformBy(camera_to_tag_0.inverse())
            field_to_camera_1 = field_to_tag_pose.transformBy(camera_to_tag_1.inverse())
            field_to_camera_pose_0 = Pose3d(field_to_camera_0.translation(), field_to_camera_0.rotation())
            field_to_camera_pose_1 = Pose3d(field_to_camera_1.translation(), field_to_camera_1.rotation())

            # Return result
            return CameraPoseObservation(
                tag_ids, field_to_camera_pose_0, errors[0][0], field_to_camera_pose_1, errors[1][0]
            )

        # Multi-tag, return one pose
        else:
            # Run SolvePNP with all tags
            try:
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    numpy.array(object_points),
                    numpy.array(image_points),
                    config_store.local_config.camera_matrix,
                    config_store.local_config.distortion_coefficients,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            except:
                return None

            # Calculate WPILib camera pose
            camera_to_field_pose = openCvPoseToWpilib(tvecs[0], rvecs[0])
            camera_to_field = Transform3d(camera_to_field_pose.translation(), camera_to_field_pose.rotation())
            field_to_camera = camera_to_field.inverse()
            field_to_camera_pose = Pose3d(field_to_camera.translation(), field_to_camera.rotation())

            # Return result
            return CameraPoseObservation(tag_ids, field_to_camera_pose, errors[0][0], None, None)
