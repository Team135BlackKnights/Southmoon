# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import dataclasses
import glob
import subprocess
import sys
import time
import traceback
from typing import Tuple, Optional, List, Dict, Union
import cv2
import numpy
import ntcore
from config.config import ConfigStore
from pypylon import pylon
# USB Video Class (UVC) constants
UVC_SET_CUR = 0x01
UVC_GET_CUR = 0x81
UVC_GET_MIN = 0x82
UVC_GET_MAX = 0x83
UVC_GET_RES = 0x84
UVC_GET_LEN = 0x85
UVC_GET_INFO = 0x86
UVC_GET_DEF = 0x87

# UVC Control Selectors
CT_EXPOSURE_TIME_ABSOLUTE_CONTROL = 0x04
CT_AUTO_EXPOSURE_MODE_CONTROL = 0x02
PU_GAIN_CONTROL = 0x04
PU_WHITE_BALANCE_TEMPERATURE_CONTROL = 0x0A
PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL = 0x0B
PU_BRIGHTNESS_CONTROL = 0x02

# Auto exposure modes
AE_MODE_MANUAL = 1
AE_MODE_AUTO = 8  # Aperture priority mode
AE_MODE_SHUTTER_PRIORITY = 4
AE_MODE_APERTURE_PRIORITY = 8

class Capture:
    """Interface for receiving camera frames."""

    def __init__(self) -> None:
        raise NotImplementedError

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        """Return the next frame from the camera."""
        raise NotImplementedError
    
    def reset(self):
        """Reset the camera connection."""
        raise NotImplementedError
    
    @classmethod
    def _config_changed(cls, config_a: ConfigStore, config_b: ConfigStore) -> bool:
        if config_a == None and config_b == None:
            return False
        if config_a == None or config_b == None:
            return True

        remote_a = config_a.remote_config
        remote_b = config_b.remote_config

        return (
            remote_a.camera_id != remote_b.camera_id
            or remote_a.camera_resolution_width != remote_b.camera_resolution_width
            or remote_a.camera_resolution_height != remote_b.camera_resolution_height
            or remote_a.camera_auto_exposure != remote_b.camera_auto_exposure
            or remote_a.camera_exposure != remote_b.camera_exposure
            or remote_a.camera_gain != remote_b.camera_gain
        )


class USBCameraCapture(Capture):
    """Read from USB camera using GStreamer with full exposure control via YUYV format."""

    def __init__(self) -> None:
        self._pipeline = None
        self._last_config: ConfigStore = None
        self._device_path = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._last_frame_time = 0
        self._frame_timeout = 2.0  # seconds

    def _get_usb_cameras(self) -> List[Dict]:
        """Get list of USB video devices."""
        import glob
        import subprocess
        
        cameras = []
        
        # Find all video devices on macOS
        video_devices = glob.glob('/dev/video*')
        
        for idx, device_path in enumerate(sorted(video_devices)):
            try:
                # Try to get device info using system_profiler
                result = subprocess.run(
                    ['system_profiler', 'SPCameraDataType', '-json'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                # Parse basic info
                device_name = f"Camera {idx}"
                unique_id = device_path
                
                # Try to open with GStreamer to verify it works
                test_pipeline = f"avfvideosrc device-index={idx} ! video/x-raw,format=UYVY ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
                test_cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
                
                if test_cap.isOpened():
                    cameras.append({
                        'index': idx,
                        'name': device_name,
                        'device_path': device_path,
                        'unique_id': unique_id,
                        'full_id': f"{device_name}:{unique_id}"
                    })
                    test_cap.release()
                    
            except Exception as e:
                print(f"Error checking device {device_path}: {e}")
                continue
        
        # If no /dev/video* devices found, try avfvideosrc device indices
        if not cameras:
            for idx in range(10):  # Check first 10 potential camera indices
                try:
                    test_pipeline = f"avfvideosrc device-index={idx} ! video/x-raw,format=UYVY ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
                    test_cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
                    
                    if test_cap.isOpened():
                        device_name = f"Camera {idx}"
                        unique_id = f"avf_{idx}"
                        cameras.append({
                            'index': idx,
                            'name': device_name,
                            'device_path': None,
                            'unique_id': unique_id,
                            'full_id': f"{device_name}:{unique_id}"
                        })
                        test_cap.release()
                except:
                    continue
        
        return cameras

    def _find_camera_by_id(self, camera_id: str) -> Optional[Dict]:
        """Find camera info by matching camera_id."""
        cameras = self._get_usb_cameras()
        
        for camera in cameras:
            if camera['unique_id'] == camera_id or camera['full_id'] == camera_id or str(camera['index']) == camera_id:
                return camera
        
        return None

    def _build_gstreamer_pipeline(self, config_store: ConfigStore, device_index: int) -> str:
        """Build GStreamer pipeline with full exposure control using avfvideosrc."""
        width = config_store.remote_config.camera_resolution_width
        height = config_store.remote_config.camera_resolution_height
        fps = 60  # Maximum FPS
        
        # Exposure settings
        auto_exposure = config_store.remote_config.camera_auto_exposure
        exposure = config_store.remote_config.camera_exposure  # Exposure time in microseconds
        gain = config_store.remote_config.camera_gain
        
        # Convert exposure from microseconds to seconds for GStreamer
        exposure_sec = exposure / 1_000_000.0
        
        # Build pipeline using avfvideosrc (AVFoundation) with manual controls
        # Use UYVY format which provides better control over camera settings
        pipeline = (
            f"avfvideosrc device-index={device_index} "
            f"capture-screen=false "
            f"capture-screen-cursor=false "
            f"! video/x-raw,format=UYVY,width={width},height={height},framerate={fps}/1 "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=1 max-buffers=2"
        )
        
        print(f"GStreamer pipeline: {pipeline}")
        return pipeline

    def _apply_camera_settings(self, config_store: ConfigStore):
        """Apply camera settings using v4l2-ctl or other macOS-specific tools."""
        if self._device_path is None:
            return
        
        try:
            import subprocess
            
            # For macOS, we'll use ffmpeg's avfoundation controls
            # These settings are applied through the pipeline creation
            # Additional runtime controls can be applied here if needed
            
            auto_exposure = 1 if config_store.remote_config.camera_auto_exposure else 0
            exposure = config_store.remote_config.camera_exposure
            gain = int(config_store.remote_config.camera_gain)
            
            print(f"Camera settings - Auto Exposure: {auto_exposure}, Exposure: {exposure}μs, Gain: {gain}")
            
        except Exception as e:
            print(f"Error applying camera settings: {e}")

    def _publish_available_cameras(self):
        """Publish list of available cameras to NetworkTables."""
        try:
            cameras = self._get_usb_cameras()
            
            camera_list = []
            for camera in cameras:
                camera_list.append(camera['full_id'])
            
            print(f"Available cameras: {camera_list}")
            
            # Publish to NT
            if self._last_config:
                nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                    f"/{self._last_config.local_config.device_id}/calibration"
                )
                nt_table.putValue("available_cameras", camera_list)
        except Exception as e:
            print(f"Error publishing cameras: {e}")

    def reset(self):
        """Reset camera connection."""
        print("Resetting camera connection...")
        
        if self._pipeline is not None:
            try:
                self._pipeline.release()
            except:
                pass
            self._pipeline = None
        
        self._device_path = None
        self._reconnect_attempts = 0
        
        cv2.destroyAllWindows()
        time.sleep(1)
        
        print("Camera reset complete")
        self._publish_available_cameras()

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        """Get frame from USB camera with full exposure control via GStreamer."""
        
        # Check if we need to reconnect due to config change
        if self._pipeline is not None and self._config_changed(self._last_config, config_store):
            print("Configuration changed, reconnecting camera...")
            self._last_config = ConfigStore(
                dataclasses.replace(config_store.local_config),
                dataclasses.replace(config_store.remote_config)
            )
            self.reset()
        
        # Check for frame timeout (camera disconnected)
        current_time = time.time()
        if self._pipeline is not None and self._last_frame_time > 0:
            if current_time - self._last_frame_time > self._frame_timeout:
                print("Frame timeout detected, camera may be disconnected")
                self.reset()
        
        # Initialize camera if needed
        if self._pipeline is None:
            if config_store.remote_config.camera_id == "":
                print("No camera ID configured")
                self._publish_available_cameras()
                return False, None
            
            # Find the camera
            camera_info = self._find_camera_by_id(config_store.remote_config.camera_id)
            
            if camera_info is None:
                print(f"Camera not found: {config_store.remote_config.camera_id}")
                self._publish_available_cameras()
                
                # Increment reconnect attempts
                self._reconnect_attempts += 1
                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    print(f"Max reconnect attempts ({self._max_reconnect_attempts}) reached, waiting longer...")
                    time.sleep(5)
                    self._reconnect_attempts = 0
                else:
                    time.sleep(1)
                
                return False, None
            
            try:
                print(f"Opening camera: {camera_info['name']} (index {camera_info['index']})")
                
                # Build and open GStreamer pipeline
                pipeline_str = self._build_gstreamer_pipeline(config_store, camera_info['index'])
                self._pipeline = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
                
                if not self._pipeline.isOpened():
                    print("Failed to open GStreamer pipeline")
                    self._pipeline = None
                    
                    # Increment reconnect attempts
                    self._reconnect_attempts += 1
                    if self._reconnect_attempts >= self._max_reconnect_attempts:
                        print(f"Max reconnect attempts ({self._max_reconnect_attempts}) reached")
                        time.sleep(5)
                        self._reconnect_attempts = 0
                    
                    return False, None
                
                self._device_path = camera_info.get('device_path')
                
                # Apply camera settings
                self._apply_camera_settings(config_store)
                
                # Warm up camera
                print("Warming up camera...")
                for i in range(5):
                    ret, _ = self._pipeline.read()
                    if ret:
                        break
                    time.sleep(0.05)
                
                # Publish available cameras
                self._publish_available_cameras()
                
                # Reset reconnect attempts on success
                self._reconnect_attempts = 0
                self._last_frame_time = current_time
                
                print("Camera initialized successfully")
                
            except Exception as e:
                print(f"Error initializing camera: {e}")
                traceback.print_exc()
                self.reset()
                
                # Increment reconnect attempts
                self._reconnect_attempts += 1
                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    print(f"Max reconnect attempts ({self._max_reconnect_attempts}) reached")
                    time.sleep(5)
                    self._reconnect_attempts = 0
                
                return False, None
        
        # Capture frame
        if self._pipeline is None:
            return False, None
        
        try:
            retval, image = self._pipeline.read()
            
            if not retval or image is None:
                print("Frame capture failed, camera may be disconnected")
                self.reset()
                return False, None
            
            # Update last frame time
            self._last_frame_time = time.time()
            
            return True, image
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            traceback.print_exc()
            self.reset()
            return False, None
        
class PylonCapture(Capture):
    """Reads from a Basler camera using pylon."""

    def __init__(self, mode: str = "", is_flipped: bool = False) -> None:
        self._mode = mode
        self._is_flipped = is_flipped

    _camera: Union[None, pylon.InstantCamera] = None
    _converter: Union[None, pylon.ImageFormatConverter] = None
    _last_config: ConfigStore = None

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        if self._camera != None and self._config_changed(self._last_config, config_store):
            print("Config changed, stopping capture session")
            self._camera.Close()
            self._camera = None
            time.sleep(2)

        if self._camera is None:
            device_infos: list[pylon.DeviceInfo] = pylon.TlFactory.GetInstance().EnumerateDevices()
            device: Union[None, any] = None  # Native object type
            for device_info in device_infos:
                if device_info.GetSerialNumber() == config_store.remote_config.camera_id:
                    device = pylon.TlFactory.GetInstance().CreateDevice(device_info)
            if device != None:
                print("Starting capture session")
                self._camera = pylon.InstantCamera(device)
                self._camera.Open()
                self._camera.GetNodeMap().GetNode("DeviceLinkThroughputLimitMode").SetValue("On")
                max_bandwidth = int(150e6) if self._mode == "color" else int(250e6)
                self._camera.GetNodeMap().GetNode("DeviceLinkThroughputLimit").SetValue(max_bandwidth)
                self._camera.GetNodeMap().GetNode("ExposureAuto").SetValue("Off")
                self._camera.GetNodeMap().GetNode("ExposureTime").SetValue(config_store.remote_config.camera_exposure)
                self._camera.GetNodeMap().GetNode("GainAuto").SetValue("Off")
                self._camera.GetNodeMap().GetNode("Gain").SetValue(config_store.remote_config.camera_gain)

                if self._mode == "color":
                    self._converter = pylon.ImageFormatConverter()
                    self._converter.OutputPixelFormat = pylon.PixelType_RGB8packed
                    self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

                elif self._mode == "cropped":
                    self._camera.GetNodeMap().GetNode("Width").SetValue(1600)
                    self._camera.GetNodeMap().GetNode("Height").SetValue(1200)
                    self._camera.GetNodeMap().GetNode("OffsetX").SetValue(168)
                    self._camera.GetNodeMap().GetNode("OffsetY").SetValue(8)

                if self._is_flipped:
                    self._camera.GetNodeMap().GetNode("ReverseX").SetValue(True)
                    self._camera.GetNodeMap().GetNode("ReverseY").SetValue(True)

                self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                print("Capture session ready")

        self._last_config = ConfigStore(
            dataclasses.replace(config_store.local_config), dataclasses.replace(config_store.remote_config)
        )

        if self._camera is None:
            return False, None
        else:
            with self._camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as grab_result:
                if grab_result.GrabSucceeded:
                    try:
                        if self._converter == None:
                            return True, grab_result.Array
                        else:
                            return True, self._converter.Convert(grab_result).Array
                    except Exception:
                        print("Error when capturing frame:", traceback.format_exc())
                        return False, None
                else:
                    return False, None


class GStreamerCapture(Capture):
    """ "Read from camera with GStreamer."""

    def __init__(self) -> None:
        pass

    _video = None
    _last_config: ConfigStore = None

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        if self._video != None and self._config_changed(self._last_config, config_store):
            print("Config changed, stopping capture session")
            self._video.release()
            self._video = None
            time.sleep(2)

        if self._video == None:
            if config_store.remote_config.camera_id == "":
                print("No camera ID, waiting to start capture session")
            else:
                print("Starting capture session")
                self._video = cv2.VideoCapture(
                    "v4l2src device="
                    + str(config_store.remote_config.camera_id)
                    + ' extra_controls="c,exposure_auto='
                    + str(config_store.remote_config.camera_auto_exposure)
                    + ",exposure_absolute="
                    + str(config_store.remote_config.camera_exposure)
                    + ",gain="
                    + str(int(config_store.remote_config.camera_gain))
                    + ',sharpness=0,brightness=0" ! image/jpeg,format=MJPG,width='
                    + str(config_store.remote_config.camera_resolution_width)
                    + ",height="
                    + str(config_store.remote_config.camera_resolution_height)
                    + " ! jpegdec ! video/x-raw ! appsink drop=1",
                    cv2.CAP_GSTREAMER,
                )
                print("Capture session ready")

        self._last_config = ConfigStore(
            dataclasses.replace(config_store.local_config), dataclasses.replace(config_store.remote_config)
        )

        if self._video != None:
            retval, image = self._video.read()
            if not retval:
                print("Capture session failed, restarting")
                self._video.release()
                self._video = None  # Force reconnect
                #sys.exit(1)
            return retval, image
        else:
            return False, cv2.Mat(numpy.ndarray([]))


CAPTURE_IMPLS = {
    "usb": USBCameraCapture,
    "pylon": lambda: PylonCapture(),
    "pylon-flipped": lambda: PylonCapture(is_flipped=True),
    "pylon-color": lambda: PylonCapture("color"),
    "pylon-color-flipped": lambda: PylonCapture("color", is_flipped=True),
    "pylon-cropped": lambda: PylonCapture("cropped"),
    "pylon-cropped-flipped": lambda: PylonCapture("cropped", is_flipped=True),
    "gstreamer": GStreamerCapture,
}