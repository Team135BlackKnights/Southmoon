import math
from typing import List, Union
import cv2
import numpy as np
from numpy.typing import NDArray

from config.config import ConfigStore
from vision_types import CameraPoseObservation, ObjDetectObservation
import pandas as pd
from wpimath.geometry import Pose3d, Translation3d, Rotation3d, Quaternion

class BlenderPoseEstimator:
    lookup_df: pd.DataFrame
    def __init__(self, blender_lookup_csv: str) -> None:
        self.lookup_df = pd.read_csv(blender_lookup_csv)
        print("[BlenderPoseEstimator] Blender lookup CSV loaded")

    def _unpack_pose3d(self, pose3d: List[float]):
        tx = pose3d[0]
        ty = pose3d[1]
        tz = pose3d[2]
        qw = pose3d[3]
        qx = pose3d[4]
        qy = pose3d[5]
        qz = pose3d[6]
        return np.array([tx, ty, tz], dtype=float), (qw, qx, qy, qz)

    @staticmethod
    def _yaw_to_quaternion(angle: float):
        half = angle / 2.0
        return math.cos(half), 0.0, 0.0, math.sin(half)

    @staticmethod
    def _multiply_quaternions(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    # Detect game pice by color
    # And create a rect around the detection
    def find_bounding_rect(self,image, lower_color, upper_color):
        """    Detects the largest contour of a specified color in the image and returns its bounding rectangle.
        Args:
            image (numpy.ndarray): The input image in which to find the contour.
            lower_color (tuple): The lower bound of the color in HSV format.
            upper_color (tuple): The upper bound of the color in HSV format.
        Returns:
            tuple: A tuple containing the coordinates (x, y) and dimensions (width, height) of the bounding rectangle.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Combine masks to get only red pixels
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Define the kernel for the morphological operation
        kernel = np.ones((7, 7), np.uint8)

        # Perform morphological opening to remove noise
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Perform morphological closing to close small holes
        # This really helps to cleanup the mask
        processed_img = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # If no contours are found, return a rectangle with all zeros
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        #draw
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return np.array([x + w//2, y + h//2, w, h], dtype=np.int32)
    def find_matching_rows(self, target, start_tol=2, max_tol=15, step=2):
        """
        Find rows in the dataframe that match the target values within a dynamically
        increasing tolerance. If no rows are found for a very small tolerance, the
        tolerance will be increased until at least one row is found or the maximum
        tolerance is reached.

        Parameters:
            df (pd.DataFrame): DataFrame with the columns.
            target (dict): Target values for each column.
            start_tol (int): Starting tolerance measured in pixels.
            max_tol (int): Maximum allowed tolerance measured in pixels.
            step (int): Increment to increase tolerance on each iteration measured in pixels.

        Returns:
            filtered_df (pd.DataFrame): DataFrame with matching rows.
            used_tol (int): Tolerance at which the matching rows were found.
        """
        if self.lookup_df is None or self.lookup_df.empty:
            return None, start_tol  # Return an empty DataFrame
        tolerance = start_tol
        while tolerance <= max_tol:
            # Create mask using np.isclose for each column
            mask = (
                np.isclose(self.lookup_df['Center_X'], target['Center_X'], atol=tolerance) &
                np.isclose(self.lookup_df['Center_Y'], target['Center_Y'], atol=tolerance) &
                np.isclose(self.lookup_df['Width'], target['Width'], atol=tolerance) &
                np.isclose(self.lookup_df['Height'], target['Height'], atol=tolerance)
            )
            filtered_df = self.lookup_df[mask]
            
            # Check if any row is found
            if not filtered_df.empty:
                return filtered_df, tolerance
            
            # Increase the tolerance and try again
            tolerance += step

        # No rows found within maximum tolerance
        print("No rows found within the max tolerance.")
        return self.lookup_df.iloc[[]], tolerance  # Return an empty DataFrame

    def find_oriented_angle(self, contour, image: NDArray[np.uint8] | None = None): 
        """
        Given a contour, find the two longest straight lines,
        average their directions, and return the dominant angle in degrees.
        """
        if len(contour) < 2:
            return None, image

        # 1. Simplify contour (remove small jitter)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 2. Collect line segments and their angles
        lines = []
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]  # wrap around
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length > 2:  # ignore tiny edges
                angle = math.degrees(math.atan2(dy, dx))
                lines.append((length, angle, p1, p2))

        if len(lines) < 2:
            return None, image

        # 3. Take two longest lines
        lines.sort(reverse=True, key=lambda x: x[0])
        longest = lines[:2]

        # 4. Compute average direction
        # Handle circular mean (avoid averaging 179° and -179° to get 0°)
        angles = [math.radians(l[1]) for l in longest]
        x_mean = np.mean([math.cos(a) for a in angles])
        y_mean = np.mean([math.sin(a) for a in angles])
        avg_angle = math.degrees(math.atan2(y_mean, x_mean))

        avg_angle = (avg_angle + 360 + 90) % 180  # normalize to [0, 180)
        # 5. Draw if can
        if image is not None:
            for _, angle, p1, p2 in longest:
                cv2.line(image, tuple(p1), tuple(p2), (0, 255, 0), 2)
                # Draw the averaged orientation line at the contour center
                cx, cy = np.mean(approx[:, 0, :], axis=0).astype(int)
                length = 50
                x2 = int(cx + length * math.cos(math.radians(avg_angle)))
                y2 = int(cy + length * math.sin(math.radians(avg_angle)))
                cv2.arrowedLine(image, (cx, cy), (x2, y2), (255, 255, 0), 2, tipLength=0.2)
        return avg_angle, image
    def find_oriented_bounding_rect(self, image, lower_color, upper_color):
        """    Detects the largest contour of a specified color in the image and returns its bounding rectangle.
        Args:
            image (numpy.ndarray): The input image in which to find the contour.
            lower_color (tuple): The lower bound of the color in HSV format.
            upper_color (tuple): The upper bound of the color in HSV format.
        Returns:
            tuple: A tuple containing the coordinates (x, y) and dimensions (width, height) of the bounding rectangle.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Combine masks to get only red pixels
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Define the kernel for the morphological operation
        kernel = np.ones((7, 7), np.uint8)
        #may need to become binary mask?
        # Perform morphological opening to remove noise
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Perform morphological closing to close small holes
        # This really helps to cleanup the mask
        processed_img = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, image

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        angle, image = self.find_oriented_angle(contour, image)

        return ((x + w / 2, y + h / 2), (w, h), angle), image
    def find_oriented_matching_rows(self, target, start_tol=2, max_tol=15, step=2):
        """
        Find rows in the dataframe that match the target values within a dynamically
        increasing tolerance. If no rows are found for a very small tolerance, the
        tolerance will be increased until at least one row is found or the maximum
        tolerance is reached.

        Parameters:
            df (pd.DataFrame): DataFrame with the columns.
            target (dict): Target values for each column.
            start_tol (int): Starting tolerance measured in pixels.
            max_tol (int): Maximum allowed tolerance measured in pixels.
            step (int): Increment to increase tolerance on each iteration measured in pixels.

        Returns:
            filtered_df (pd.DataFrame): DataFrame with matching rows.
            used_tol (int): Tolerance at which the matching rows were found.
        """
        
        if self.lookup_df is None or self.lookup_df.empty:
            return None, start_tol
        tolerance = start_tol
        while tolerance <= max_tol:
            # Create mask using np.isclose for each column
            mask = (
                np.isclose(self.lookup_df['Center_X'], target['Center_X'], atol=tolerance) &
                np.isclose(self.lookup_df['Center_Y'], target['Center_Y'], atol=tolerance) &
                np.isclose(self.lookup_df['Width'], target['Width'], atol=tolerance) &
                np.isclose(self.lookup_df['Height'], target['Height'], atol=tolerance) &
                (np.isclose(self.lookup_df['Image_Angle'], target['Image_Angle'], atol=tolerance) |
                    np.isclose(self.lookup_df['Image_Angle'], (target['Image_Angle'] + 180) % 360, atol=tolerance) |
                    np.isclose(self.lookup_df['Image_Angle'], (target['Image_Angle'] - 180) % 360, atol=tolerance))
            )
            filtered_df = self.lookup_df[mask]
            
            # Check if any row is found
            if not filtered_df.empty:
                return filtered_df, tolerance
            
            # Increase the tolerance and try again
            tolerance += step

        # No rows found within maximum tolerance
        print("No rows found within the max tolerance.")
        return self.lookup_df.iloc[[]], tolerance  # Return an empty DataFrame


    def _match_position_from_rect(
        self,
        config: ConfigStore,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        oriented_angle: float | None,
    ):
        """Resolve the best dataframe match for a detected rectangle."""

        if config.local_config.obj_use_oriented_detection and oriented_angle is not None:
            target = {
                'Center_X': center_x,
                'Center_Y': center_y,
                'Width': width,
                'Height': height,
                'Image_Angle': oriented_angle,
            }
            filtered_df, _ = self.find_oriented_matching_rows(target, start_tol=25, max_tol=90, step=3)
            if filtered_df.empty:
                return None
            return [
                float(filtered_df['X_Position'].mean()),
                float(filtered_df['Y_Position'].mean()),
                math.radians(float(filtered_df['Angle'].mean())),
            ]

        target = {
            'Center_X': center_x,
            'Center_Y': center_y,
            'Width': width,
            'Height': height,
        }
        filtered_df, _ = self.find_matching_rows(target, start_tol=25, max_tol=40, step=3)
        if filtered_df.empty:
            return None
        return [
            float(filtered_df['X_Position'].mean()),
            float(filtered_df['Y_Position'].mean()),
        ]


    def estimate_position(
        self,
        image: np.ndarray,
        config: ConfigStore,
    ):
        """
        Estimate the position of any object within an image based on color detection alone.
        """
        #target res is like (1600,1304)
        #image is cv2 image of 1600,1304 color format BGR ?
        #df is dataframe with columns Center_X, Center_Y, Width, Height, Angle
        lower_hsv = tuple(config.remote_config.lower_hsv)
        upper_hsv = tuple(config.remote_config.upper_hsv)
        rect_info, debug_image = self.find_oriented_bounding_rect(image, lower_hsv, upper_hsv)
        if rect_info is None:
            return None, debug_image, 'No contour'
        (center_x, center_y), (width, height), oriented_angle = rect_info
        if oriented_angle is None and config.local_config.obj_use_oriented_detection:
            return None, debug_image, 'No orientation'
        if self.lookup_df is None or self.lookup_df.empty:
            return None, debug_image, 'No lookup data'
        result = self._match_position_from_rect(
            config=config,
            center_x=float(center_x),
            center_y=float(center_y),
            width=float(width),
            height=float(height),
            oriented_angle=oriented_angle,
        )
        if result is None:
            return None, debug_image, 'No picture in tolerance'
        return result, debug_image, ''     
    def estimate_ai_position(
        self,
        image_observation: ObjDetectObservation, 
        config_store: ConfigStore
    ):
        if self.lookup_df is None or self.lookup_df.empty:
            return None, 'No lookup data'
        corners = np.asarray(image_observation.corner_pixels, dtype=np.float32)
        if corners.size == 0:
            return None, 'No corners'

        points = corners.reshape(-1, 2)
        if points.shape[0] < 2:
            return None, 'Not enough points'

        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        center_x = float((min_xy[0] + max_xy[0]) / 2.0)
        center_y = float((min_xy[1] + max_xy[1]) / 2.0)
        width = float(max_xy[0] - min_xy[0])
        height = float(max_xy[1] - min_xy[1])
        oriented_angle: float | None = None

        if points.shape[0] >= 3:
            contour = points.reshape(-1, 1, 2).astype(np.int32)
            oriented_angle, _ = self.find_oriented_angle(contour)
        result = self._match_position_from_rect(
            config=config_store,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            oriented_angle=oriented_angle,
        )
        if result is None:
            return None,'No picture in tolerance'
        return result, ''
    def position_to_field_pose(
        self,
        position: dict,
        config: ConfigStore,
        debug: str,
    ):
        """
        Convert a detected position to a field pose observation.
        """
        try:
            out: dict = {}
            p0_t, p0_q = self._unpack_pose3d(config.remote_config.field_camera_pose)
            rel_yaw = float(position[2]) if len(position) > 2 else 0.0
            rel_q = self._yaw_to_quaternion(rel_yaw)
            field_q = self._multiply_quaternions(p0_q, rel_q)
            pose = Pose3d(
                Translation3d(position[0] + p0_t[0], position[1] + p0_t[1], 0.051),
                Rotation3d(Quaternion(field_q[0], field_q[1], field_q[2], field_q[3])),
            )
            print(f"Distance to object: x={position[0]:.2f} m, y={position[1]:.2f} m, angle={rel_yaw:.2f} rad")
            print (f"Estimated field pose: x={pose.translation.x:.2f}, y={pose.translation.y:.2f}, yaw={pose.rotation.toEulerAngles()[2]:.2f} rad")
            
            return CameraPoseObservation(
                tag_ids= [-1],
                pose_0=pose,
                pose_1=None,
                error_0=0.0,
                error_1=None,
                ), debug
        except Exception:
            return None, debug + "\nPose serialization failed."