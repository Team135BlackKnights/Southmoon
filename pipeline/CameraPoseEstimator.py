# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from typing import List, Union

import cv2
import math
import numpy
from config.config import ConfigStore
from pipeline.coordinate_systems import openCvPoseToWpilib, wpilibTranslationToOpenCv
from vision_types import CameraPoseObservation, FiducialImageObservation, ObjDetectObservation
from wpimath.geometry import *
import ntcore


class CameraPoseEstimator:
    def __init__(self) -> None:
        raise NotImplementedError

    def solve_camera_pose(
        self, image_observations: List[FiducialImageObservation], config_store: ConfigStore
    ) -> Union[CameraPoseObservation, None]:
        raise NotImplementedError

class MultiBumperCameraPoseEstimator(CameraPoseEstimator):    
    def __init__(self, bumper_size_m: float = 0.8382, bottom_z: float = 0.0, top_z: float = 0.1778):
        self.bumper_size_m = bumper_size_m
        self.bottom_z = bottom_z
        self.top_z = top_z


    def _unpack_pose3d(self, pose3d: List[float]):
        #todo
        tx = pose3d[0]
        ty = pose3d[1]
        tz = pose3d[2]
        qw = pose3d[3]
        qx = pose3d[4]
        qy = pose3d[5]
        qz = pose3d[6]
        return numpy.array([tx, ty, tz], dtype=float), (qw, qx, qy, qz)

    def _quat_to_rotmat(self, q):
        qw, qx, qy, qz = q
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if n == 0:
            raise ValueError("zero quaternion???")
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        R = numpy.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )
        return R

    def _rotmat_to_quat(self, R):
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
        tr = m00 + m11 + m22
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m21 - m12) / S
            qy = (m02 - m20) / S
            qz = (m10 - m01) / S
        elif (m00 > m11) and (m00 > m22):
            S = math.sqrt(1.0 + m00 - m11 - m22) * 2
            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S
        elif m11 > m22:
            S = math.sqrt(1.0 + m11 - m00 - m22) * 2
            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S
        else:
            S = math.sqrt(1.0 + m22 - m00 - m11) * 2
            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S
        return (qw, qx, qy, qz)

    # Intersect a camera ray with a horizontal plane z = plane_z (both in field frame)
    def _intersect_ray_with_z(self, cam_pos_field, dir_field, plane_z, eps=1e-8):
        dz = dir_field[2]
        if abs(dz) < eps:
            return None, None

        t = (plane_z - cam_pos_field[2]) / dz
        Ps = cam_pos_field + t * dir_field
        return Ps, t

    def solve_camera_pose(self, image_observations: List[ObjDetectObservation], config_store: ConfigStore) -> tuple [Union[CameraPoseObservation, None], str]:
        debug_msgs = []

        if len(image_observations) == 0:
            debug_msgs.append("NA IO")
            return None, "\n".join(debug_msgs)

        # Exit if no field pos available
        if config_store.remote_config.field_camera_pose == None:
            debug_msgs.append("NA POSE")
            return None, "\n".join(debug_msgs)

        # camera pose in field frame (field -> camera)
        cam_field_pose = config_store.remote_config.field_camera_pose
        K = numpy.array(config_store.local_config.camera_matrix, dtype=float)
        dist_coeffs = config_store.local_config.distortion_coefficients
        dist = numpy.array(dist_coeffs, dtype=float) if dist_coeffs is not None else None
        if len(cam_field_pose) != 7:
            debug_msgs.append("NA POSE LEN")
            return None, "\n".join(debug_msgs)
        cam_pos_field, cam_quat = self._unpack_pose3d(cam_field_pose)
        WPI_TO_CV = numpy.array(
            [[0, -1, 0],
             [0,  0, -1],
             [1,  0, 0]],
            dtype=float,
        )
        CV_TO_WPI = WPI_TO_CV.T

        # NB: field_camera_pose publishes camera->field rotation in WPILib frame.
        R_camera_field_wpi = self._quat_to_rotmat(cam_quat)
        R_field_camera_wpi = R_camera_field_wpi.T
        debug_msgs.append(f"CAM POS: {cam_pos_field}")
        debug_msgs.append(f"CAM QUAT: {cam_quat}")

        # Pre-compute bumper corner model in the bumper object frame (origin at bottom center)
        half_width = self.bumper_size_m / 2.0
        bumper_height = self.top_z - self.bottom_z
        object_points_wpilib = numpy.array(
            [
                [0.0, half_width, bumper_height],   # top-left
                [0.0, -half_width, bumper_height],  # top-right
                [0.0, half_width, 0.0],             # bottom-left
                [0.0, -half_width, 0.0],            # bottom-right
            ],
            dtype=float,
        )
        object_points_cv = numpy.array(
            [wpilibTranslationToOpenCv(Translation3d(*pt)) for pt in object_points_wpilib],
            dtype=float,
        )

        results = []
        errs = []
        
        for obs_idx, obs in enumerate(image_observations):
            if obs.corner_pixels is None or len(obs.corner_pixels) != 4:
                debug_msgs.append(f"OBS {obs_idx}: BAD CORNERS")
                continue
            image_points = numpy.array(obs.corner_pixels, dtype=float)

            try:
                solve_ok, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                    object_points_cv,
                    image_points,
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            except cv2.error as e:
                debug_msgs.append(f"OBS {obs_idx}: solvePnP ERR {e}")
                continue

            if not solve_ok or rvecs is None or len(rvecs) == 0:
                debug_msgs.append(f"OBS {obs_idx}: NO SOL")
                continue

            debug_msgs.append(f"OBS {obs_idx}: SOLUTIONS {len(rvecs)}")

            for sol_idx, (rvec, tvec, err_val) in enumerate(zip(rvecs, tvecs, reproj)):
                R_oc_cv, _ = cv2.Rodrigues(rvec)
                R_oc_wpi = CV_TO_WPI @ R_oc_cv @ WPI_TO_CV
                t_co_wpi = CV_TO_WPI @ tvec.reshape(3)

                R_of_wpi = R_camera_field_wpi @ R_oc_wpi
                quat = self._rotmat_to_quat(R_of_wpi)

                object_pos_field = cam_pos_field + R_camera_field_wpi @ t_co_wpi
                field_to_object_pose = Pose3d(
                    Translation3d(*object_pos_field),
                    Rotation3d(Quaternion(quat[0], quat[1], quat[2], quat[3])),
                )

                distance = float(numpy.linalg.norm(t_co_wpi))
                roll = field_to_object_pose.rotation().X()
                pitch = field_to_object_pose.rotation().Y()
                yaw = field_to_object_pose.rotation().Z()
                err_scalar = float(err_val[0] if numpy.ndim(err_val) else err_val)

                debug_msgs.append(
                    f"  SOL {sol_idx}: ERR={err_scalar:.2f}px DIST={distance:.3f}m RPY=({roll:.3f},{pitch:.3f},{yaw:.3f})"
                )

                results.append((field_to_object_pose, err_scalar, obs.obj_class if hasattr(obs, "obj_class") else obs_idx))
                errs.append(err_scalar)

        if not results:
            debug_msgs.append("NO RESULTS")
            return None, "\n".join(debug_msgs)

        best_idx = int(numpy.argmin(errs))
        best_pose, best_err, best_id = results[best_idx]
        debug_msgs.append(f"BEST IDX: {best_idx} ERR: {best_err:.2f}")
        return (
            CameraPoseObservation(
                tag_ids=[best_id],
                pose_0=best_pose,
                error_0=float(best_err),
                pose_1=None,
                error_1=None,
            ),
            "\n".join(debug_msgs),
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
