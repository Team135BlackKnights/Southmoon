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

import math
import itertools
from typing import List, Union

import cv2
import numpy

class MultiBumperCameraPoseEstimator(CameraPoseEstimator):
    def __init__(self, bumper_size_m: float = 0.8382, bottom_z: float = 0.0, top_z: float = 0.1778):
        self.bumper_size_m = bumper_size_m
        self.bottom_z = bottom_z
        self.top_z = top_z

    def _unpack_pose3d(self, pose3d: List[float]):
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

    # ---------------- geometry helpers ----------------

    def _undistort_normalized(self, pts_px, K, dist):
        pts = numpy.array(pts_px, dtype=float).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, K, dist, P=None)
        return und.reshape(-1, 2)

    def _ray_dir_camera_from_normalized(self, norm_xy):
        norm_xy = numpy.array(norm_xy, dtype=float)
        return numpy.column_stack((norm_xy[:, 0], norm_xy[:, 1], numpy.ones(len(norm_xy), dtype=float)))

    def _intersect_ray_with_z(self, cam_pos_field, dir_field, plane_z, eps=1e-8):
        dz = dir_field[2]
        if abs(dz) < eps:
            return None, None
        t = (plane_z - cam_pos_field[2]) / dz
        P = cam_pos_field + t * dir_field
        return P, t

    def _umeyama_rigid_transform(self, src_pts, dst_pts):
        src = numpy.array(src_pts, dtype=float)
        dst = numpy.array(dst_pts, dtype=float)
        assert src.shape == dst.shape and src.shape[0] >= 1

        n = src.shape[0]
        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        src_c = src - mu_src
        dst_c = dst - mu_dst

        cov = (dst_c.T @ src_c) / n
        U, S, Vt = numpy.linalg.svd(cov)
        D = numpy.identity(3)
        if numpy.linalg.det(U @ Vt) < 0:
            D[2, 2] = -1.0
        R = U @ D @ Vt
        t = mu_dst - R @ mu_src
        return R, t

    def _two_top_point_length_fit(self, cam_pos_field, ray_dirs_field, real_width, bumper_height):
        """
        ray_dirs_field: list of 2 normalized direction vectors (Nx3)
        real_width: known physical distance between top corners (m)
        bumper_height: known vertical distance from top to bottom (m)
        """
        if len(ray_dirs_field) != 2:
            raise ValueError("Need exactly two ray directions")

        r0 = ray_dirs_field[0] / numpy.linalg.norm(ray_dirs_field[0])
        r1 = ray_dirs_field[1] / numpy.linalg.norm(ray_dirs_field[1])
        C = cam_pos_field

        # Solve for lambda0, lambda1 so that |(C + λ1 r1) - (C + λ0 r0)| = real_width
        # Define: d = C + λ1 r1 - (C + λ0 r0) = λ1 r1 - λ0 r0
        # We also know (r0·r1) = cos(theta). This gives a single equation in λ1−λ0 relation.
        cos_theta = numpy.dot(r0, r1)
        A = 1 - cos_theta**2
        if abs(A) < 1e-6:
            # nearly parallel rays, fallback
            return None, None

        # geometric derivation for lambda0, lambda1 that minimizes |λ1 r1 - λ0 r0|^2 - W^2 = 0
        # Let r0·r0 = r1·r1 = 1
        # The optimal symmetric solution is λ0 = λ1 = λ (approx mid-depth)
        # but we can compute using: W^2 = λ^2 * |r1 - r0|^2 ⇒ λ = W / |r1 - r0|
        diff = r1 - r0
        norm_diff = numpy.linalg.norm(diff)
        if norm_diff < 1e-6:
            return None, None

        lam = real_width / norm_diff
        P0 = C + lam * r0
        P1 = C + lam * r1

        # Compute midpoint of top edge in field coords
        T_mid = 0.5 * (P0 + P1)

        # Drop down by bumper height along field +Z
        O_field = T_mid - numpy.array([0, 0, bumper_height], dtype=float)

        # Compute yaw from the XY vector between P1 and P0
        v_xy = P1[:2] - P0[:2]
        yaw = math.atan2(v_xy[1], v_xy[0])
        R_yaw = numpy.array([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw),  math.cos(yaw), 0.0],
            [0.0,            0.0,           1.0],
        ])

        t = O_field - (R_yaw @ numpy.array([0.0, 0.0, 0.0]))
        return R_yaw, t


    # ---------------- main solver ----------------

    def solve_camera_pose(self, image_observations: List[ObjDetectObservation], config_store: ConfigStore) -> tuple[Union[CameraPoseObservation, None], str]:
        debug_msgs = []

        if len(image_observations) == 0:
            debug_msgs.append("NA IO")
            return None, "\n".join(debug_msgs)

        if config_store.remote_config.field_camera_pose is None:
            debug_msgs.append("NA POSE")
            return None, "\n".join(debug_msgs)

        cam_field_pose = config_store.remote_config.field_camera_pose
        K = numpy.array(config_store.local_config.camera_matrix, dtype=float)
        dist_coeffs = config_store.local_config.distortion_coefficients
        dist = numpy.zeros((5,), dtype=float) if dist_coeffs is None else numpy.array(dist_coeffs, dtype=float)

        if len(cam_field_pose) != 7:
            debug_msgs.append("NA POSE LEN")
            return None, "\n".join(debug_msgs)

        cam_pos_field, cam_quat = self._unpack_pose3d(cam_field_pose)

        # CV ↔ WPI converters (same as your previous code)
        WPI_TO_CV = numpy.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
        CV_TO_WPI = WPI_TO_CV.T

        # R_camera_field_wpi maps camera-frame vectors (WPILib camera axes) into field-frame vectors
        R_camera_field_wpi = self._quat_to_rotmat(cam_quat)

        debug_msgs.append(f"CAM POS: {cam_pos_field}")
        debug_msgs.append(f"CAM QUAT: {cam_quat}")

        # model points in object frame (origin bottom center)
        half_width = self.bumper_size_m / 2.0
        bumper_height = self.top_z - self.bottom_z
        model_pts = numpy.array([
            [0.0,  half_width, bumper_height],   # top-left
            [0.0, -half_width, bumper_height],   # top-right
            [0.0,  half_width, 0.0],             # bottom-left
            [0.0, -half_width, 0.0],             # bottom-right
        ], dtype=float)

        results = []
        errs = []

        for obs_idx, obs in enumerate(image_observations):
            if obs.corner_pixels is None:
                debug_msgs.append(f"OBS {obs_idx}: NO CORNERS")
                continue

            px_list = obs.corner_pixels
            pts_px = []
            valid_model_indices = []
            for i, p in enumerate(px_list):
                if p is None:
                    continue
                pts_px.append((float(p[0]), float(p[1])))
                valid_model_indices.append(i)

            if len(pts_px) == 0:
                debug_msgs.append(f"OBS {obs_idx}: NO VALID PIXELS")
                continue

            # undistort to normalized coords (these are in OpenCV camera frame)
            norm_xy = self._undistort_normalized(pts_px, K, dist)  # Nx2
            dirs_cv = self._ray_dir_camera_from_normalized(norm_xy)  # Nx3 (OpenCV camera axes)

            # Convert dirs from CV camera axes -> WPILib camera axes (IMPORTANT)
            dirs_wpi_cam = (CV_TO_WPI @ dirs_cv.T).T  # Nx3

            # Rotate camera-frame directions into field-frame directions
            dirs_field = (R_camera_field_wpi @ dirs_wpi_cam.T).T  # Nx3

            observed_field_points = []
            ray_ts = []
            valid_indices = []
            for local_idx, model_idx in enumerate(valid_model_indices):
                # absolute model corner Z in field (assume object origin sits at ground z=0)
                model_corner_z = model_pts[model_idx][2] + self.bottom_z
                P, tval = self._intersect_ray_with_z(cam_pos_field, dirs_field[local_idx], model_corner_z)
                if P is None:
                    continue
                # sanity: ignore intersections behind camera (tval < 0)
                if tval is not None and tval <= 0:
                    continue
                observed_field_points.append(P)
                ray_ts.append(tval)
                valid_indices.append(local_idx)

            num_valid = len(valid_indices)
            if num_valid == 0:
                debug_msgs.append(f"OBS {obs_idx}: NO RAY INTERSECTS")
                continue

            src_all = numpy.array([model_pts[valid_model_indices[i]] for i in valid_indices], dtype=float)
            dst_all = numpy.array(observed_field_points, dtype=float)

            # subset enumeration (cheap for up to 4)
            best_solution = None
            best_residual = float("inf")
            if num_valid >= 3:
                min_required = 3
            elif num_valid == 2:
                min_required = 2
            else:
                min_required = 1

            for k in range(min_required, num_valid + 1):
                for subset in itertools.combinations(range(num_valid), k):
                    sub_src = src_all[list(subset), :]
                    sub_dst = dst_all[list(subset), :]

                    try:
                        if k >= 3:
                            R_est, t_est = self._umeyama_rigid_transform(sub_src, sub_dst)
                        elif len(valid_model_indices) == 2:
                            # === TOP TWO CORNERS ONLY ===
                            ray_dirs_for_these = [dirs_field[local_idx] for local_idx in valid_indices]
                            R_est, t_est = self._two_top_point_length_fit(
                                cam_pos_field,
                                ray_dirs_for_these,
                                self.bumper_size_m,  # known real width
                                0.05,                # known height
                            )
                            if R_est is None:
                                continue  # fallback

                            # --- Stupid distance bias correction for 2-corner case ---
                            APPLY_BIAS_FOR_K_EQ_2 = True
                            BIAS_METERS = 15 
                            MIN_DISTANCE = 0.1      # safety lower bound

                            if APPLY_BIAS_FOR_K_EQ_2:
                                # Compute current object position & distance
                                obj_origin_field = (R_est @ numpy.array([0.0, 0.0, 0.0])) + t_est
                                orig_dist = float(numpy.linalg.norm(cam_pos_field - obj_origin_field))

                                corrected_dist = orig_dist - BIAS_METERS
                                if corrected_dist < MIN_DISTANCE:
                                    corrected_dist = MIN_DISTANCE

                                # Shift along camera→object vector to match corrected distance
                                dir_vec = obj_origin_field - cam_pos_field
                                dir_norm = numpy.linalg.norm(dir_vec)
                                if dir_norm > 1e-8:
                                    dir_unit = dir_vec / dir_norm
                                    obj_origin_field = cam_pos_field + dir_unit * corrected_dist
                                    t_est = obj_origin_field - (R_est @ numpy.array([0.0, 0.0, 0.0]))

                                    debug_msgs.append(
                                        f"    [K==2 BIAS] dist {orig_dist:.3f} → {corrected_dist:.3f} (bias={BIAS_METERS})"
                                    )
                                else:
                                    debug_msgs.append("    [K==2 BIAS] skipped (degenerate direction)")

                        else:
                            R_est = numpy.eye(3, dtype=float)
                            t_est = sub_dst[0] - sub_src[0]
                    except Exception:
                        continue

                    pred = (R_est @ src_all.T).T + t_est
                    residuals = numpy.linalg.norm(pred - dst_all, axis=1)
                    mean_res = float(numpy.mean(residuals))
                    if mean_res < best_residual:
                        best_residual = mean_res
                        best_solution = (R_est, t_est, residuals, subset)

            if best_solution is None:
                debug_msgs.append(f"OBS {obs_idx}: NO VALID TRANSFORM")
                continue

            R_est, t_est, residuals, used_subset = best_solution
            obj_origin_field = (R_est @ numpy.array([0.0, 0.0, 0.0])) + t_est

            quat = self._rotmat_to_quat(R_est)
            field_to_object_pose = Pose3d(
                Translation3d(*obj_origin_field),
                Rotation3d(Quaternion(quat[0], quat[1], quat[2], quat[3])),
            )

            err_scalar = float(best_residual)
            distance = float(numpy.linalg.norm(cam_pos_field - obj_origin_field))
            debug_msgs.append(f"  OBS{obs_idx}: USED={used_subset} MEAN_RES={err_scalar:.4f} DIST={distance:.3f}")
            results.append((field_to_object_pose, err_scalar, obs.obj_class if hasattr(obs, "obj_class") else obs_idx))
            errs.append(err_scalar)

        if not results:
            debug_msgs.append("NO RESULTS")
            return None, "\n".join(debug_msgs)

        best_idx = int(numpy.argmin(errs))
        best_pose, best_err, best_id = results[best_idx]
        debug_msgs.append(f"BEST IDX: {best_idx} ERR: {best_err:.4f}")
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
