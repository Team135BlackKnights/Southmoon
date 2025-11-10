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
    def __init__(self, bumper_size_m: float = 0.8382, bottom_z: float = 0.0381, top_z: float = 0.1778):
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
            return None  # parallel, cooked
        t = (plane_z - cam_pos_field[2]) / dz
        if t <= 0:
            # veryyyy likely behind camera, aka... tf?
            return None
        return cam_pos_field + t * dir_field

    def solve_camera_pose(self, image_observations: List[ObjDetectObservation], config_store: ConfigStore) -> tuple:
        import traceback
        debug_msgs = []
        try:
            if len(image_observations) == 0:
                debug_msgs.append("NA IO")
                return None, "\n".join(debug_msgs)

            if config_store.remote_config.field_camera_pose is None:
                debug_msgs.append("NA POSE")
                return None, "\n".join(debug_msgs)

            cam_field_pose = config_store.remote_config.field_camera_pose
            K = numpy.array(config_store.local_config.camera_matrix, dtype=float)
            Kinv = numpy.linalg.inv(K)

            if len(cam_field_pose) != 7:
                debug_msgs.append("NA POSE LEN")
                return None, "\n".join(debug_msgs)

            cam_pos_field, cam_quat = self._unpack_pose3d(cam_field_pose)
            debug_msgs.append(f"CAM POS: {cam_pos_field}")
            debug_msgs.append(f"CAM QUAT: {cam_quat}")

            # Build rotation(s) defensively
            try:
                R_camera_field = self._quat_to_rotmat(cam_quat)
            except Exception as e:
                debug_msgs.append(f"ERROR building R_camera_field from cam_quat: {e}")
                debug_msgs.append(traceback.format_exc())
                # Try a fallback quaternion ordering (common cause)
                try:
                    q_shifted = (cam_quat[3], cam_quat[0], cam_quat[1], cam_quat[2])
                    R_camera_field = self._quat_to_rotmat(q_shifted)
                    debug_msgs.append("Succeeded with shifted quaternion ordering (fallback).")
                except Exception as e2:
                    debug_msgs.append(f"Fallback shift also failed: {e2}")
                    debug_msgs.append(traceback.format_exc())
                    return None, "\n".join(debug_msgs)

            # Diagnostics: forward vectors
            try:
                fwd_plusz = R_camera_field @ numpy.array([0.0, 0.0, 1.0])
                fwd_minusz = R_camera_field @ numpy.array([0.0, 0.0, -1.0])
                fwd_plusz_t = R_camera_field.T @ numpy.array([0.0, 0.0, 1.0])
                debug_msgs.append(f"FWD (R @ [0,0,1]) = {fwd_plusz}")
                debug_msgs.append(f"FWD (R @ [0,0,-1]) = {fwd_minusz}")
                debug_msgs.append(f"FWD (R.T @ [0,0,1]) = {fwd_plusz_t}")
            except Exception as e:
                debug_msgs.append(f"FWD diag failed: {e}")
                debug_msgs.append(traceback.format_exc())

            # Candidate transform factory (explicit not using closures that might capture mutated values)
            def cand_R(d): return R_camera_field @ d
            def cand_RT(d): return R_camera_field.T @ d
            def cand_RT_negZ(d): 
                dd = d.copy()
                dd[2] = -dd[2]
                return R_camera_field.T @ dd

            # Also try shifted quaternion rotations if we built it earlier
            try:
                q_shifted = (cam_quat[3], cam_quat[0], cam_quat[1], cam_quat[2])
                R_shifted = self._quat_to_rotmat(q_shifted)
                def cand_Rs(d): return R_shifted @ d
                def cand_RsT(d): return R_shifted.T @ d
            except Exception:
                R_shifted = None
                cand_Rs = None
                cand_RsT = None

            candidates = [
                (cand_R, "R @ d_cam"),
                (cand_RT, "R.T @ d_cam"),
                (cand_RT_negZ, "R.T @ (d_cam with Z neg)"),
            ]
            if cand_Rs is not None:
                candidates += [(cand_Rs, "R_shifted @ d_cam"), (cand_RsT, "R_shifted.T @ d_cam")]

            debug_msgs.append("CANDIDATES: " + ", ".join([name for (_, name) in candidates]))

            # helper: score candidate using first viable observation
            def score_candidate(candidate_fn, obs):
                score = 0
                total = 0
                if obs is None or obs.corner_pixels is None:
                    return 0, 0
                for uv in obs.corner_pixels:
                    try:
                        u, v = float(uv[0]), float(uv[1])
                        uv1 = numpy.array([u, v, 1.0], dtype=float)
                        d_cam = Kinv @ uv1
                        d_field = candidate_fn(d_cam)
                        if d_field[2] < -1e-6:
                            score += 1
                        total += 1
                    except Exception:
                        # ignore per-ray failures for scoring
                        total += 1
                return score, total

            # Choose first usable observation to pick candidate
            first_obs = None
            for obs in image_observations:
                if obs and obs.corner_pixels is not None and len(obs.corner_pixels) >= 1:
                    first_obs = obs
                    break

            selected_candidate = None
            selected_name = None
            if first_obs is not None:
                best = (-1, None, None)
                for fn, name in candidates:
                    try:
                        s, tot = score_candidate(fn, first_obs)
                    except Exception as e:
                        debug_msgs.append(f"Candidate scoring failed for {name}: {e}")
                        debug_msgs.append(traceback.format_exc())
                        s, tot = 0, 0
                    debug_msgs.append(f"CAND SCORE {name}: {s}/{tot}")
                    if s > best[0]:
                        best = (s, fn, name)
                if best[0] > 0:
                    selected_candidate = best[1]
                    selected_name = best[2]
                    debug_msgs.append(f"SELECTED: {selected_name} with score {best[0]}")
            if selected_candidate is None:
                selected_candidate = cand_RT
                selected_name = "fallback R.T @ d_cam (default)"
                debug_msgs.append("No good candidate found — using fallback R.T @ d_cam")

            results = []
            errs = []

            # main loop — each observation processed defensively
            for obs_idx, obs in enumerate(image_observations):
                try:
                    if obs.corner_pixels is None or len(obs.corner_pixels) != 4:
                        debug_msgs.append(f"OBS {obs_idx}: BAD CORNERS")
                        continue

                    corner_zs = [self.top_z, self.top_z, self.bottom_z, self.bottom_z]
                    corner_world_pts = []
                    bad = False
                    debug_msgs.append(f"OBS {obs_idx}: INTERSECTING (using {selected_name})")

                    # center ray diagnostic
                    try:
                        uv_center = numpy.array([K[0, 2], K[1, 2], 1.0])
                        d_center_cam = Kinv @ uv_center
                        d_center_field = selected_candidate(d_center_cam)
                        debug_msgs.append(f"  CENTER d_cam={d_center_cam} -> d_field={d_center_field}")
                    except Exception:
                        debug_msgs.append("  CENTER diag failed")
                        debug_msgs.append(traceback.format_exc())

                    for corner_idx, (uv, plane_z) in enumerate(zip(obs.corner_pixels, corner_zs)):
                        try:
                            u, v = float(uv[0]), float(uv[1])
                            uv1 = numpy.array([u, v, 1.0], dtype=float)
                            d_cam = Kinv @ uv1
                            d_field = selected_candidate(d_cam)
                            debug_msgs.append(f"  C{corner_idx}: uv=({u:.1f},{v:.1f}) d_cam=({d_cam[0]:.3f},{d_cam[1]:.3f},{d_cam[2]:.3f}) d_field=({d_field[0]:.3f},{d_field[1]:.3f},{d_field[2]:.3f})")
                            P = self._intersect_ray_with_z(cam_pos_field, d_field, plane_z)
                            if P is None:
                                dz = d_field[2]
                                if abs(dz) < 1e-8:
                                    debug_msgs.append(f"  C{corner_idx}: PARALLEL (dz={dz:.6f})")
                                else:
                                    t = (plane_z - cam_pos_field[2]) / dz
                                    debug_msgs.append(f"  C{corner_idx}: BEHIND CAM (t={t:.6f})")
                                bad = True
                                break
                            corner_world_pts.append(P)
                            debug_msgs.append(f"  C{corner_idx}: OK P=({P[0]:.3f},{P[1]:.3f},{P[2]:.3f})")
                        except Exception as e:
                            debug_msgs.append(f"  C{corner_idx}: per-corner exception: {e}")
                            debug_msgs.append(traceback.format_exc())
                            bad = True
                            break

                    if bad:
                        debug_msgs.append(f"OBS {obs_idx}: FAILED INTERSECTION")
                        continue

                    corner_world_pts = numpy.vstack(corner_world_pts)
                    centroid = numpy.mean(corner_world_pts, axis=0)
                    v_x = corner_world_pts[1] - corner_world_pts[0]
                    v_y = corner_world_pts[3] - corner_world_pts[0]

                    def norm(v):
                        n = numpy.linalg.norm(v)
                        return v / n if n > 1e-9 else v

                    debug_msgs.append(f"OBS {obs_idx}: NORMING")
                    try:
                        x_axis = norm(v_x)
                        y_axis = norm(v_y - numpy.dot(v_y, x_axis) * x_axis)
                        z_axis = numpy.cross(x_axis, y_axis)
                        z_axis = norm(z_axis)
                    except Exception as e:
                        debug_msgs.append(f"OBS {obs_idx}: norming exception: {e}")
                        debug_msgs.append(traceback.format_exc())
                        continue

                    R_obj_to_field = numpy.column_stack([x_axis, y_axis, z_axis])
                    quat = self._rotmat_to_quat(R_obj_to_field)
                    field_to_object_pose = Pose3d(
                        Translation3d(centroid[0], centroid[1], centroid[2]),
                        Rotation3d(Quaternion(quat[0], quat[1], quat[2], quat[3])),
                    )
                    debug_msgs.append(f"OBS {obs_idx}: REPROG")

                    reproj_err = 0.0
                    for i in range(4):
                        try:
                            P_field = corner_world_pts[i]
                            R_field_camera = R_camera_field.T
                            p_cam = R_field_camera @ (P_field - cam_pos_field)
                            if p_cam[2] <= 0:
                                reproj_err += 1e6
                                continue
                            proj = K @ (p_cam / p_cam[2])
                            u_proj, v_proj = proj[0], proj[1]
                            du = u_proj - float(obs.corner_pixels[i][0])
                            dv = v_proj - float(obs.corner_pixels[i][1])
                            reproj_err += math.hypot(du, dv)
                        except Exception as e:
                            debug_msgs.append(f"  reproj corner {i} exception: {e}")
                            debug_msgs.append(traceback.format_exc())
                            reproj_err += 1e6
                    reproj_err /= 4.0

                    results.append((field_to_object_pose, corner_world_pts))
                    errs.append(reproj_err)
                    debug_msgs.append(f"OBS {obs_idx}: SUCCESS err={reproj_err:.2f}")
                except Exception as e:
                    debug_msgs.append(f"OBS {obs_idx}: outer exception: {e}")
                    debug_msgs.append(traceback.format_exc())
                    continue

            if len(results) == 0:
                debug_msgs.append("NO RESULTS")
                return None, "\n".join(debug_msgs)

            best_idx = int(numpy.argmin(errs))
            best_pose, _ = results[best_idx]
            best_err = float(errs[best_idx])

            return (
                CameraPoseObservation(
                    tag_ids=[0],
                    pose_0=best_pose,
                    error_0=best_err,
                    pose_1=None,
                    error_1=None,
                ),
                "\n".join(debug_msgs),
            )

        except Exception as e_main:
            # Catch anything that escaped
            debug_msgs.append(f"UNHANDLED EXCEPTION IN solve_camera_pose: {e_main}")
            try:
                debug_msgs.append(traceback.format_exc())
            except Exception:
                debug_msgs.append("failed to format traceback")
            # Return gracefully with debug messages so you can see what happened
            return None, "\n".join(debug_msgs)


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
