# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import dataclasses
import os
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Tuple, Optional, List, Dict, Union
import usb.core
import usb.util
import AVFoundation
import cv2
import numpy
import ntcore
from config.config import ConfigStore
from pypylon import pylon
import traceback
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
UV_ID_RE = re.compile(r"usb_([0-9a-fA-F]{4})_([0-9a-fA-F]{4})(?:_([0-9A-Za-z_]+))?")

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
    """macOS-only capture class that uses a small IOKit helper to set UVC controls,
    then captures frames via avfvideosrc (GStreamer) at 60 FPS.
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_config: Optional[ConfigStore] = None
        self._device_info: Optional[Dict] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        self._frame_timeout = .1
        self._last_frame_time = 0.0
        # location of the helper; change if you put it elsewhere
        self.uvc_helper_path = "/Users/pennrobotics/Documents/GitHub/Southmoon/uvc_helper"

    # -------------------------
    # Probe avfvideosrc indices and return camera list
    # -------------------------

    def _get_usb_cameras(self) -> List[Dict]:
        """Get list of USB video devices with unique identifiers."""
        cameras = []
        
        # Find all USB devices that might be cameras
        devices = usb.core.find(find_all=True)
        
        video_device_index = 0
        for dev in devices:
            try:
                # Check if this is a video device
                is_video = False
                
                # Check device class
                if dev.bDeviceClass == 0x0E or dev.bDeviceClass == 0xEF:  # Video or Miscellaneous
                    is_video = True
                
                # Check interface classes for UVC
                if not is_video:
                    try:
                        for cfg in dev:
                            for intf in cfg:
                                if intf.bInterfaceClass == 0x0E:  # Video class
                                    is_video = True
                                    break
                            if is_video:
                                break
                    except:
                        pass
                
                if is_video:
                    # Get device info
                    vendor_id = dev.idVendor
                    product_id = dev.idProduct
                    
                    # Try to get serial number
                    serial = None
                    try:
                        if dev.iSerialNumber:
                            serial = usb.util.get_string(dev, dev.iSerialNumber)
                    except:
                        pass
                    
                    # Try to get product name
                    product_name = f"Camera {video_device_index}"
                    try:
                        if dev.iProduct:
                            product_name = usb.util.get_string(dev, dev.iProduct)
                    except:
                        pass
                    
                    # Create unique ID based on VID/PID and serial
                    if serial:
                        unique_id = f"usb_{vendor_id:04x}_{product_id:04x}_{serial}"
                    else:
                        # Fall back to bus/address if no serial
                        unique_id = f"usb_{vendor_id:04x}_{product_id:04x}_{dev.bus:03d}_{dev.address:03d}"
                    
                    cameras.append({
                        'index': video_device_index,
                        'name': product_name,
                        'vendor_id': vendor_id,
                        'product_id': product_id,
                        'serial': serial,
                        'bus': dev.bus,
                        'address': dev.address,
                        'unique_id': unique_id,
                        'full_id': f"{product_name}:{unique_id}",
                        'device': dev  # Keep reference to USB device
                    })
                    
                    video_device_index += 1
                    
            except Exception as e:
                continue
        
        return cameras

    def _find_camera_by_id(self, camera_id: str) -> Optional[Dict]:
        """Find camera info by matching camera_id."""
        cameras = self._get_usb_cameras()
        
        for camera in cameras:
            if camera['unique_id'] == camera_id or camera['full_id'] == camera_id:
                return camera
        
        return None
        

    # -------------------------
    # Parse your camera ID strings like:
    # "SPCA2630 PC Camera:usb_05c8_0a00_001_001"
    # return dict {vendor, product, serial}
    # -------------------------
    def _parse_usb_id(self, camera_id: str) -> Optional[Dict]:
        if not camera_id:
            return None
        m = UV_ID_RE.search(camera_id)
        if not m:
            return None
        vid_s, pid_s, serial = m.groups()
        return {
            "vendor_id": int(vid_s, 16),
            "product_id": int(pid_s, 16),
            "serial": serial
        }

    # -------------------------
    # Call the uvc_helper helper to set AE/exposure/gain (best-effort)
    # -------------------------
    def _call_uvc_helper(self, vendor_id: int, product_id: int, serial: Optional[str],
                         exposure_us: Optional[int], gain: Optional[int],
                         manual: bool = False) -> bool:
        if not os.path.exists(self.uvc_helper_path):
            print(f"uvc_helper not found at {self.uvc_helper_path}")
            return False

        cmd = [self.uvc_helper_path]
        cmd += ["--vendor", f"0x{vendor_id:04x}", "--product", f"0x{product_id:04x}"]
        if serial:
            cmd += ["--serial", serial]
        if exposure_us is not None:
            cmd += ["--exposure", str(int(exposure_us))]
        if gain is not None:
            cmd += ["--gain", str(int(gain))]
        if manual:
            cmd += ["--manual"]

        try:
            print(f"Running uvc_helper: {' '.join(cmd)}")
            # Run with elevated permissions if available — uvc_helper may need root
            # We don't escalate here automatically; caller should run the main process with sudo or the helper itself will fail and print diagnostics
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            print("uvc_helper stdout:\n", proc.stdout)
            if proc.stderr:
                print("uvc_helper stderr:\n", proc.stderr)
            return proc.returncode == 0
        except Exception as e:
            print("Error running uvc_helper:", e)
            return False

    # -------------------------
    # Build avfvideosrc GStreamer pipeline (60fps)
    # -------------------------
    def _build_pipeline(self, device_index: int, width: int, height: int) -> str:
        return (
            f"avfvideosrc device-index={device_index} capture-screen=false capture-screen-cursor=false ! video/x-raw,format=UYVY,width={width},height={height},framerate=60/1 ! videoconvert ! video/x-raw,format=UYVY ! appsink drop=1 max-buffers=2"

        )

    # -------------------------
    # Apply exposure/gain via helper; try to find device index first
    # -------------------------
    def _apply_settings_and_open(self, config_store: ConfigStore) -> bool:
        cam_id = config_store.remote_config.camera_id
        parsed = self._parse_usb_id(cam_id)
        # If no USB pattern, fallback to simple index (user may supply "Camera 0:avf_0")
        device_index = None
        try:
            # if camera_id contains "avf_<n>" or index
            avf_m = re.search(r"avf_(\d+)", cam_id or "")
            if avf_m:
                device_index = int(avf_m.group(1))
        except:
            device_index = None

        # If parsed vendor/product available, call helper before opening capture
        if parsed:
            success = self._call_uvc_helper(parsed["vendor_id"], parsed["product_id"], parsed.get("serial"),
                                            int(config_store.remote_config.camera_exposure),
                                            int(config_store.remote_config.camera_gain),
                                            manual=not bool(config_store.remote_config.camera_auto_exposure))
            if not success:
                print("uvc_helper reported failure or partial success — continuing to open capture (settings may not be applied).")

        # If device_index still None, probe 0..9 for an available index and pick one.
        if device_index is None:
            cams = self._get_usb_cameras()
            if not cams:
                print("No avfvideosrc-compatible devices found.")
                return False
            # take first available
            device_index = cams[0]["index"]

        width = int(config_store.remote_config.camera_resolution_width)
        height = int(config_store.remote_config.camera_resolution_height)
        pipeline = self._build_pipeline(device_index, width, height)
        print("Opening pipeline:", pipeline)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        # Small warm-up
        if not self._cap or not self._cap.isOpened():
            print("Failed to open avfvideosrc pipeline.")
            if self._cap:
                try:
                    self._cap.release()
                except:
                    pass
            self._cap = None
            return False

        for i in range(8):
            ret, _ = self._cap.read()
            if ret:
                break
            time.sleep(0.05)

        return True

    # -------------------------
    # Reset
    # -------------------------
    def reset(self):
        if self._cap:
            try:
                self._cap.release()
            except:
                pass
            self._cap = None
        self._device_info = None
        time.sleep(0.5)

    # -------------------------
    # Config changed test (simple)
    # -------------------------
    def _config_changed(self, old: Optional[ConfigStore], new: ConfigStore) -> bool:
        if old is None:
            return True
        r_old = old.remote_config
        r_new = new.remote_config
        keys = ["camera_resolution_width","camera_resolution_height","camera_auto_exposure","camera_exposure","camera_gain","camera_id"]
        for k in keys:
            if getattr(r_old, k) != getattr(r_new, k):
                return True
        return False

    # -------------------------
    # Public: get_frame
    # -------------------------
    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, Optional[Any]]:
        # Reconnect on config change
        if self._cap is not None and self._config_changed(self._last_config, config_store):
            print("Config changed — reconnecting.")
            self.reset()

        if self._cap is None:
            ok = self._apply_settings_and_open(config_store)
            if not ok:
                # publish cameras if needed
                try:
                    self._publish_available_cameras()
                except:
                    pass
                return False, None
            self._last_config = dataclasses.replace(config_store)

        # read frame
        try:
            ret, img = self._cap.read()
            if not ret or img is None:
                print("Frame read failed — resetting capture")
                self.reset()
                return False, None
            self._last_frame_time = time.time()
            return True, img
        except Exception as e:
            print("Exception reading frame:", e)
            traceback.print_exc()
            self.reset()
            return False, None

    def _publish_available_cameras(self):
        cams = self._get_usb_cameras()
        camera_list = [c["full_id"] for c in cams]
        print("Available cameras:", camera_list)
        if self._last_config:
            try:
                nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                    f"/{self._last_config.local_config.device_id}/calibration"
                )
                nt_table.putValue("available_cameras", camera_list)
            except Exception:
                pass

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