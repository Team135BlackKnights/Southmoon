# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import dataclasses
import os
import sys
import time
import traceback
from typing import Tuple, Optional, List, Dict, Union
import subprocess
import shutil
import usb.core
import usb.util
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
            or remote_a.camera_saturation != remote_b.camera_saturation
            or remote_a.camera_hue != remote_b.camera_hue
            or remote_a.camera_white_balance != remote_b.camera_white_balance
            or remote_a.camera_auto_white_balance != remote_b.camera_auto_white_balance
        )


class USBCameraCapture(Capture):
    """Read from USB camera with direct UVC control for exposure and other settings."""

    def __init__(self) -> None:
        self._cv_capture = None
        self._usb_device = None
        self._device_index = None
        self._last_config: ConfigStore = None
        self._camera_controls = {}
        self._ffmpeg_proc = None
        self._frame_bytes = None
        self._width = None
        self._height = None

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
                    if vendor_id == 0x05c8:
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
                    else:
                        #TODO: remove this else after testing
                        '''cameras.append({
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
                    
                        video_device_index += 1'''
                    
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

    def _uvc_get_control(self, device, unit_id: int, control: int, length: int) -> Optional[bytes]:
        """Get UVC control value."""
        try:
            # Find video control interface
            intf_num = 0
            for cfg in device:
                for intf in cfg:
                    if intf.bInterfaceClass == 0x0E and intf.bInterfaceSubClass == 0x01:
                        intf_num = intf.bInterfaceNumber
                        break
            
            data = device.ctrl_transfer(
                bmRequestType=0xA1,  # Class specific, interface, device to host
                bRequest=UVC_GET_CUR,
                wValue=(control << 8),
                wIndex=(unit_id << 8) | intf_num,
                data_or_wLength=length
            )
            return data
        except Exception as e:
            print(f"Failed to get control {control:02x}: {e}")
            return None

    def _uvc_set_control(self, device, unit_id: int, control: int, data: bytes) -> bool:
        """Set UVC control value."""
        try:
            # Find video control interface
            intf_num = 0
            for cfg in device:
                for intf in cfg:
                    if intf.bInterfaceClass == 0x0E and intf.bInterfaceSubClass == 0x01:
                        intf_num = intf.bInterfaceNumber
                        break
            
            device.ctrl_transfer(
                bmRequestType=0x21,  # Class specific, interface, host to device
                bRequest=UVC_SET_CUR,
                wValue=(control << 8),
                wIndex=(unit_id << 8) | intf_num,
                data_or_wLength=data
            )
            return True
        except Exception as e:
            print(f"Failed to set control {control:02x}: {e}")
            return False

    def _find_control_units(self, device) -> Dict[str, int]:
        """Find UVC control unit IDs."""
        units = {}
        
        # Common unit IDs (these vary by camera)
        # Try common ranges
        for unit_id in range(1, 10):
            # Try to read exposure to test if this is a camera terminal
            data = self._uvc_get_control(device, unit_id, CT_EXPOSURE_TIME_ABSOLUTE_CONTROL, 4)
            if data is not None:
                units['camera_terminal'] = unit_id
                print(f"Found camera terminal at unit {unit_id}")
                break
        
        for unit_id in range(1, 10):
            # Try to read gain to test if this is a processing unit
            data = self._uvc_get_control(device, unit_id, PU_GAIN_CONTROL, 2)
            if data is not None:
                units['processing_unit'] = unit_id
                print(f"Found processing unit at unit {unit_id}")
                break
        
        return units

    def _apply_uvc_settings(self, config_store: ConfigStore):
        """Apply camera settings via direct UVC controls."""
        if self._usb_device is None:
            return
        
        # On macOS, AVFoundation owns the device and TCC blocks direct UVC controls for most apps.
        # Rely on OpenCV/AVFoundation property controls instead to avoid EACCES and no-op behavior.
        if sys.platform == "darwin":
            print("macOS detected: skipping direct UVC controls and using AVFoundation/OpenCV props")
            self._apply_opencv_settings(config_store)
            return
        
        try:
            # Find control units if not cached
            if 'units' not in self._camera_controls:
                self._camera_controls['units'] = self._find_control_units(self._usb_device)
            
            units = self._camera_controls['units']
            
            if 'camera_terminal' in units:
                ct_unit = units['camera_terminal']
                
                # Set auto exposure mode
                if config_store.remote_config.camera_auto_exposure:
                    ae_mode = AE_MODE_AUTO
                else:
                    ae_mode = AE_MODE_MANUAL
                
                ae_data = ae_mode.to_bytes(1, byteorder='little')
                if self._uvc_set_control(self._usb_device, ct_unit, CT_AUTO_EXPOSURE_MODE_CONTROL, ae_data):
                    print(f"Set auto exposure mode to {ae_mode}")
                
                # Set exposure time (in 100μs units for UVC)
                if not config_store.remote_config.camera_auto_exposure:
                    # Convert from our units to UVC units (100μs)
                    exposure_100us = int(config_store.remote_config.camera_exposure / 100)
                    exp_data = exposure_100us.to_bytes(4, byteorder='little')
                    
                    if self._uvc_set_control(self._usb_device, ct_unit, CT_EXPOSURE_TIME_ABSOLUTE_CONTROL, exp_data):
                        print(f"Set exposure to {exposure_100us * 100}μs")
            
            if 'processing_unit' in units:
                pu_unit = units['processing_unit']
                
                # Set gain
                gain_value = int(config_store.remote_config.camera_gain)
                gain_data = gain_value.to_bytes(2, byteorder='little')
                
                if self._uvc_set_control(self._usb_device, pu_unit, PU_GAIN_CONTROL, gain_data):
                    print(f"Set gain to {gain_value}")
        
        except Exception as e:
            print(f"Error applying UVC settings: {e}")
            # Fall back to OpenCV settings
            self._apply_opencv_settings(config_store)

    def _apply_opencv_settings(self, config_store: ConfigStore):
        """Fallback: Apply settings via OpenCV where possible."""
        if self._cv_capture is None:
            return
        
        try:
            self._apply_ns_iokit_settings(config_store)
            
        except Exception as e:
            print(f"Error applying OpenCV settings: {e}")

    def _apply_ns_iokit_settings(self, config_store: ConfigStore):
        """macOS: Use bundled ns-iokit-ctl helper to force exposure/gain via IOKit.

        Requires building ns-iokit-ctl first. Enable with env USE_NS_IOKIT_CTL=1.
        """
        if self._usb_device is None:
            print("ns-iokit-ctl: no USB device available")
            return

        try:
            force_rebuild = os.getenv("NS_IOKIT_CTL_REBUILD", "0") == "1"
            vid = int(self._usb_device.idVendor)
            pid = int(self._usb_device.idProduct)
            # Location ID: we don't have it via libusb; pass 0 to select first match
            location_hex = "0"

            disable_auto = 0 if config_store.remote_config.camera_auto_exposure else 1
            exposure = float(config_store.remote_config.camera_exposure)
            gain = int(config_store.remote_config.camera_gain)
            saturation = int(config_store.remote_config.camera_saturation)
            hue = int(config_store.remote_config.camera_hue)
            disable_auto_white_balance = int(config_store.remote_config.camera_auto_white_balance)
            white_balance = int(config_store.remote_config.camera_white_balance)
            # Find tool binary
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            tool_dir = os.path.join(repo_root, "ns-iokit-ctl")
            build_dir = os.path.join(tool_dir, "build")
            tool_bin = os.path.join(build_dir, "ns_iokit_ctl")

            if force_rebuild and os.path.exists(build_dir):
                print("NS_IOKIT_CTL_REBUILD=1 set: removing existing build directory to force rebuild...")
                try:
                    shutil.rmtree(build_dir)
                except Exception as de:
                    print(f"Failed to remove build directory: {de}")

            if force_rebuild or not os.path.exists(tool_bin):
                print("ns-iokit-ctl build required; building via CMake...")
                try:
                    os.makedirs(build_dir, exist_ok=True)
                    # Configure
                    subprocess.run(["cmake", "-S", tool_dir, "-B", build_dir], check=True)
                    # Build
                    subprocess.run(["cmake", "--build", build_dir, "--config", "Release"], check=True)
                    #remove the flag
                    if force_rebuild:
                        os.environ["NS_IOKIT_CTL_REBUILD"] = "0"
                except Exception as be:
                    print(f"Failed to build ns-iokit-ctl: {be}")
                    return

            if not os.path.exists(tool_bin):
                print("ns-iokit-ctl binary not found after build")
                return

            args = [
                tool_bin,
                f"{vid:04x}",
                f"{pid:04x}",
                location_hex,
                str(disable_auto),
                str(exposure),
                str(gain),
                str(saturation),
                str(hue),
                str(disable_auto_white_balance),
                str(white_balance)
            ]
            print("ns-iokit-ctl: ", " ".join(args))
            try:
                result = subprocess.run(args, capture_output=True, text=True, timeout=5)
                if result.stdout:
                    print(result.stdout.strip())
                if result.stderr:
                    print(result.stderr.strip())
                if result.returncode != 0:
                    print(f"ns-iokit-ctl failed with code {result.returncode}")
            except Exception as re:
                print(f"Error running ns-iokit-ctl: {re}")
        except Exception as e:
            print(f"ns-iokit-ctl error: {e}")

    def _publish_available_cameras(self):
        """Publish list of available cameras to NetworkTables."""
        cameras = self._get_usb_cameras()
        
        camera_list = []
        for camera in cameras:
            camera_list.append(camera['full_id'])
        print (f"Available cameras: {camera_list}")
        # Publish to NT
        if self._last_config:
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                f"/{self._last_config.local_config.device_id}/calibration"
            )
            nt_table.putValue("available_cameras", camera_list)

    def reset(self):
        """Reset camera connection."""
        if self._cv_capture is not None:
            try:
                self._cv_capture.release()
            except:
                pass
            self._cv_capture = None
        
        # Clean up FFmpeg process
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.terminate()
                self._ffmpeg_proc.wait(timeout=2)
            except:
                try:
                    self._ffmpeg_proc.kill()
                except:
                    pass
            self._ffmpeg_proc = None
        
        self._usb_device = None
        self._device_index = None
        self._camera_controls = {}
        self._frame_bytes = None
        self._width = None
        self._height = None
        
        cv2.destroyAllWindows()
        time.sleep(.05)
        print("Camera reset complete")
        self._publish_available_cameras()

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        """Get frame from USB camera with proper exposure control."""
        
        # Check if we need to reconnect
        if self._cv_capture is not None and self._config_changed(self._last_config, config_store):
            self._last_config = ConfigStore(
                dataclasses.replace(config_store.local_config),
                dataclasses.replace(config_store.remote_config)
            )
            self.reset()
        
        # Initialize camera if needed
        if self._cv_capture is None:
            if config_store.remote_config.camera_id == "":
                print("No camera ID configured")
                # Still publish available cameras
                self._publish_available_cameras()
                return False, None
            
            # Find the camera
            camera_info = self._find_camera_by_id(config_store.remote_config.camera_id)
            
            if camera_info is None:
                print(f"Camera not found: {config_store.remote_config.camera_id}")
                self._publish_available_cameras()
                return False, None
            
            try:
                # Open with OpenCV
                print(f"Opening camera at index {camera_info['index']}")
                self._cv_capture = cv2.VideoCapture(camera_info['index'])
                
                if not self._cv_capture.isOpened():
                    print("Failed to open camera")
                    self._cv_capture = None
                    return False, None
                
                # Set resolution
                self._cv_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 
                                    config_store.remote_config.camera_resolution_width)
                self._cv_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,
                                    config_store.remote_config.camera_resolution_height)
                
                # Store USB device reference
                self._usb_device = camera_info['device']
                self._device_index = camera_info['index']
                
                # Warm up camera
                print("Warming up camera...")
                for i in range(5):
                    ret, _ = self._cv_capture.read()
                    if ret:
                        break
                    time.sleep(0.01)
                
                # Apply UVC settings (with OpenCV fallback)
                self._apply_uvc_settings(config_store)
                
                # Publish available cameras
                self._publish_available_cameras()
                
                print("Camera initialized successfully")
                
            except Exception as e:
                print(f"Error initializing camera: {e}")
                traceback.print_exc()
                self.reset()
                return False, None        
        
        # Capture frame
        if self._cv_capture is None:
            return False, None
        
        try:
            retval, image = self._cv_capture.read()
            
            if not retval:
                print("Frame capture failed, resetting")
                self.reset()
                return False, None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return True, image
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
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
