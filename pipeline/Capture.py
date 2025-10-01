# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import dataclasses
import subprocess
import sys
import time
import traceback
from typing import Tuple, Optional, List, Dict, Union
import usb.core
import usb.util
import AVFoundation
import cv2
import numpy
import ntcore
from config.config import ConfigStore
from pypylon import pylon

class Capture:
    """Interface for receiving camera frames."""

    def __init__(self) -> None:
        raise NotImplementedError

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        """Return the next frame from the camera."""
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


class DefaultCapture(Capture):
    """ "Read from camera with default OpenCV config."""

    def __init__(self) -> None:
        pass

    _video = None
    _last_config: ConfigStore = None

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        if self._video != None and self._config_changed(self._last_config, config_store):
            print("Restarting capture session")
            self._video.release()
            self._video = None

        if self._video == None:
            #Log an error
            try:
                self._video = cv2.VideoCapture(config_store.local_config.device_id)
            except:
                print("Error: Camera not found for ID: " + config_store.local_config.device_id)
                sys.exit(1)
            self._video.set(cv2.CAP_PROP_FRAME_WIDTH, config_store.remote_config.camera_resolution_width)
            self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, config_store.remote_config.camera_resolution_height)
            self._video.set(cv2.CAP_PROP_AUTO_EXPOSURE, config_store.remote_config.camera_auto_exposure)
            self._video.set(cv2.CAP_PROP_EXPOSURE, config_store.remote_config.camera_exposure)
            self._video.set(cv2.CAP_PROP_GAIN, int(config_store.remote_config.camera_gain))

        # FIXED: Create a deep copy of the config
        self._last_config = ConfigStore(
            dataclasses.replace(config_store.local_config), 
            dataclasses.replace(config_store.remote_config)
        )

        retval, image = self._video.read()
        return retval, image
class USBCameraCapture:
    """Read from USB camera with unique USB identification on macOS."""

    def __init__(self) -> None:
        self._cv_capture = None
        self._device_index = None
        self._last_config: ConfigStore = None

    def _get_usb_cameras(self) -> List[Dict]:
        """Get list of USB video devices with unique identifiers."""
        cameras = []
        
        # Find all USB devices with Video class (0x0E) or vendor-specific video devices
        devices = usb.core.find(find_all=True)
        
        video_device_index = 0
        for dev in devices:
            try:
                # Check if this is a video device (class 0x0E or interface class 0x0E)
                is_video = False
                
                # Check device class
                if dev.bDeviceClass == 0x0E or dev.bDeviceClass == 0xEF:  # Video or Miscellaneous
                    is_video = True
                
                # Check interface classes
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
                    # Create unique ID from USB address and device identifiers
                    # Format: bus-address-vendor-product or with serial if available
                    bus = dev.bus
                    address = dev.address
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
                    
                    # Create unique ID in the format you requested
                    if serial:
                        unique_id = f"0x{bus:x}{address:02x}{vendor_id:04x}{product_id:04x}{hash(serial) & 0xFFFF:04x}"
                    else:
                        unique_id = f"0x{bus:x}{address:02x}{vendor_id:04x}{product_id:04x}"
                    
                    cameras.append({
                        'index': video_device_index,
                        'name': product_name,
                        'vendor_id': vendor_id,
                        'product_id': product_id,
                        'serial': serial,
                        'bus': bus,
                        'address': address,
                        'unique_id': unique_id,
                        'full_id': f"{product_name}:{unique_id}"
                    })
                    
                    video_device_index += 1
                    
            except Exception as e:
                # Skip devices we can't access
                continue
        
        return cameras

    def _find_device_index(self, camera_id: str) -> Optional[int]:
        """Find OpenCV device index by matching camera_id to unique_id."""
        cameras = self._get_usb_cameras()
        
        for camera in cameras:
            if camera['unique_id'] == camera_id or camera['full_id'] == camera_id:
                return camera['index']
        
        return None

    def _apply_opencv_settings(self, config_store: ConfigStore):
        """Apply camera settings via OpenCV (limited but works for some cameras)."""
        if self._cv_capture is None:
            return
        
        try:
            # Try to set auto exposure (values vary by camera)
            # 0.25 = manual mode, 0.75 = auto mode for some cameras
            if config_store.remote_config.camera_auto_exposure:
                self._cv_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                self._cv_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            # Set exposure (may or may not work depending on camera)
            if not config_store.remote_config.camera_auto_exposure:
                # OpenCV exposure is typically in log scale or device-specific units
                # Try negative values for manual mode (common on macOS)
                exposure_value = -config_store.remote_config.camera_exposure / 10.0
                self._cv_capture.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
            
            # Set gain
            gain_value = int(config_store.remote_config.camera_gain)
            self._cv_capture.set(cv2.CAP_PROP_GAIN, gain_value)
            
            # Verify what was actually set
            actual_exposure = self._cv_capture.get(cv2.CAP_PROP_EXPOSURE)
            actual_gain = self._cv_capture.get(cv2.CAP_PROP_GAIN)
            actual_auto = self._cv_capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            
            print(f"Settings applied - Auto: {actual_auto}, Exposure: {actual_exposure}, Gain: {actual_gain}")
            
        except Exception as e:
            print(f"Error applying camera settings via OpenCV: {e}")

    def _camera_id_changed(self, old_config: ConfigStore, new_config: ConfigStore) -> bool:
        """Check if camera ID or resolution changed (requires reconnect)."""
        if old_config is None:
            return True
        return (
            old_config.remote_config.camera_id != new_config.remote_config.camera_id or
            old_config.remote_config.camera_resolution_width != new_config.remote_config.camera_resolution_width or
            old_config.remote_config.camera_resolution_height != new_config.remote_config.camera_resolution_height
        )

    def _config_changed(self, old_config: ConfigStore, new_config: ConfigStore) -> bool:
        """Check if any camera settings changed."""
        if old_config is None:
            return True
        
        old_remote = old_config.remote_config
        new_remote = new_config.remote_config
        
        return (
            old_remote.camera_auto_exposure != new_remote.camera_auto_exposure or
            old_remote.camera_exposure != new_remote.camera_exposure or
            old_remote.camera_gain != new_remote.camera_gain
        )

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        """Get frame from camera."""
        
        # Reconnect if camera ID or resolution changed
        if self._cv_capture is not None and self._camera_id_changed(self._last_config, config_store):
            print("Restarting capture session (camera ID or resolution changed)")
            self._cv_capture.release()
            self._cv_capture = None

        # Initialize device if needed
        if self._cv_capture is None:
            if config_store.remote_config.camera_id == "":
                print("No camera ID, waiting to start capture session")
                return False, None
            
            # Get and publish available cameras
            cameras = self._get_usb_cameras()
            print("Available USB cameras:")
            
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                f"/{config_store.local_config.device_id}/calibration"
            )
            
            camera_list = []
            for camera in cameras:
                serial_info = f" (Serial: {camera['serial']})" if camera['serial'] else ""
                print(f"  [{camera['index']}] {camera['name']} - {camera['unique_id']}{serial_info}")
                camera_list.append(camera['full_id'])
            
            nt_table.putValue("available_cameras", camera_list)
            
            # Find matching device
            device_index = self._find_device_index(config_store.remote_config.camera_id)
            
            if device_index is None:
                print(f"Camera not found for ID: {config_store.remote_config.camera_id}")
                print("Available unique IDs:", [c['unique_id'] for c in cameras])
                return False, None
            
            try:
                # Open OpenCV capture
                print(f"Opening camera at index {device_index}")
                self._cv_capture = cv2.VideoCapture(device_index)
                self._device_index = device_index
                
                if not self._cv_capture.isOpened():
                    print(f"Failed to open camera at index {device_index}")
                    self._cv_capture = None
                    return False, None
                
                # Set resolution
                self._cv_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 
                                    config_store.remote_config.camera_resolution_width)
                self._cv_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 
                                    config_store.remote_config.camera_resolution_height)
                
                # Apply initial settings
                self._apply_opencv_settings(config_store)
                
                # Verify resolution
                actual_width = self._cv_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self._cv_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Camera opened: {actual_width}x{actual_height}")
                
                # Save config
                self._last_config = ConfigStore(
                    dataclasses.replace(config_store.local_config),
                    dataclasses.replace(config_store.remote_config),
                )
                
            except Exception as e:
                print(f"Error opening camera: {e}")
                import traceback
                traceback.print_exc()
                if self._cv_capture is not None:
                    self._cv_capture.release()
                    self._cv_capture = None
                return False, None
        
        # Apply settings if they changed (without reconnecting)
        elif self._config_changed(self._last_config, config_store):
            print("Camera settings changed, reapplying")
            self._apply_opencv_settings(config_store)
            self._last_config = ConfigStore(
                dataclasses.replace(config_store.local_config),
                dataclasses.replace(config_store.remote_config),
            )
        
        # Capture frame
        if self._cv_capture is None:
            return False, None
        
        try:
            retval, image = self._cv_capture.read()
            
            if not retval:
                print("Failed to capture frame, reconnecting...")
                self._cv_capture.release()
                self._cv_capture = None
                return False, None
            
            return True, image
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            self._cv_capture.release()
            self._cv_capture = None
            return False, None

    def __del__(self):
        """Cleanup on deletion."""
        if self._cv_capture is not None:
            self._cv_capture.release()
            
class AVFoundationCapture(Capture):
    """Read from camera with OpenCV and AVFoundation."""

    def __init__(self) -> None:
        pass

    _video = None
    _avf_device = None  # Store the actual AVFoundation device
    _last_config: ConfigStore = None

    def _apply_camera_settings(self, config_store: ConfigStore):
        """Apply exposure and gain settings via direct AVFoundation APIs."""
        if self._avf_device is None:
            print("No AVFoundation device available")
            return

        # Lock device for configuration
        error = None
        success = self._avf_device.lockForConfiguration_(error)
        if not success:
            print(f"Failed to lock device for configuration: {error}")
            return

        try:
            # Auto exposure
            if config_store.remote_config.camera_auto_exposure:
                if self._avf_device.isExposureModeSupported_(AVFoundation.AVCaptureExposureModeContinuousAutoExposure):
                    self._avf_device.setExposureMode_(AVFoundation.AVCaptureExposureModeContinuousAutoExposure)
                    print("Set to auto exposure mode")
            else:
                # Manual exposure
                if self._avf_device.isExposureModeSupported_(AVFoundation.AVCaptureExposureModeCustom):
                    duration_sec = config_store.remote_config.camera_exposure / 1000.0
                    
                    # Clamp to device min/max
                    min_duration = self._avf_device.activeFormat().minExposureDuration()
                    max_duration = self._avf_device.activeFormat().maxExposureDuration()
                    
                    duration = AVFoundation.CMTimeMakeWithSeconds(duration_sec, 1000000000)
                    
                    # Use current ISO or set a specific value
                    current_iso = self._avf_device.ISO()
                    
                    self._avf_device.setExposureModeCustomWithDuration_ISO_completionHandler_(
                        duration, current_iso, None
                    )
                    print(f"Set manual exposure: {duration_sec}s, ISO: {current_iso}")

            # Brightness (if available)
            try:
                if hasattr(self._avf_device, 'videoZoomFactor'):
                    # Some devices support brightness adjustment
                    pass
            except:
                pass

        finally:
            self._avf_device.unlockForConfiguration()

    def _camera_id_changed(self, old_config: ConfigStore, new_config: ConfigStore) -> bool:
        """Check if camera ID or resolution changed (requires reconnect)."""
        if old_config is None:
            return True
        return (
            old_config.remote_config.camera_id != new_config.remote_config.camera_id or
            old_config.remote_config.camera_resolution_width != new_config.remote_config.camera_resolution_width or
            old_config.remote_config.camera_resolution_height != new_config.remote_config.camera_resolution_height
        )

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        # Only reconnect if camera ID or resolution changed
        if self._video != None and self._camera_id_changed(self._last_config, config_store):
            print("Restarting capture session (camera ID or resolution changed)")
            self._video.release()
            self._video = None
            self._avf_device = None

        if self._video == None:
            if config_store.remote_config.camera_id == "":
                print("No camera ID, waiting to start capture session")
            else:
                devices = list(
                    AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
                )
                devices.sort(key=lambda x: x.uniqueID())
                print("Available cameras:")
                nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                    "/" + str(config_store.local_config.device_id) + "/calibration"
                )
                for device in devices:
                    print(f"  {device.localizedName()} ({device.uniqueID()})")

                # Publish available cameras to NetworkTables
                nt_table.putValue(
                    "available_cameras",
                    [f"{device.localizedName()}:{device.uniqueID()}" for device in devices],
                )

                for index, device in enumerate(devices):
                    if device.uniqueID() == str(config_store.remote_config.camera_id):
                        # Store the AVFoundation device object
                        self._avf_device = device
                        
                        # Create OpenCV capture
                        self._video = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
                        
                        # Set resolution via OpenCV
                        self._video.set(
                            cv2.CAP_PROP_FRAME_WIDTH,
                            config_store.remote_config.camera_resolution_width,
                        )
                        self._video.set(
                            cv2.CAP_PROP_FRAME_HEIGHT,
                            config_store.remote_config.camera_resolution_height,
                        )

                        # Apply camera settings via AVFoundation
                        self._apply_camera_settings(config_store)

                        # Save config
                        self._last_config = ConfigStore(
                            dataclasses.replace(config_store.local_config),
                            dataclasses.replace(config_store.remote_config),
                        )
                        break
        
        # Apply settings if exposure/gain changed (without reconnecting)
        elif self._config_changed(self._last_config, config_store):
            print("Camera settings changed, applying via AVFoundation")
            self._apply_camera_settings(config_store)
            self._last_config = ConfigStore(
                dataclasses.replace(config_store.local_config),
                dataclasses.replace(config_store.remote_config),
            )

        if self._video == None:
            if str(config_store.remote_config.camera_id) != "0":
                print("Camera not found, retrying...")
            return False, None
        else:
            retval, image = self._video.read()
            if not retval:
                print("Capture session failed, retrying...")
                self._video.release()
                self._video = None
                self._avf_device = None
            return retval, image
        
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
    "": DefaultCapture,
    "opencv": DefaultCapture,
    "usb": USBCameraCapture,
    "avfoundation": AVFoundationCapture,
    "pylon": lambda: PylonCapture(),
    "pylon-flipped": lambda: PylonCapture(is_flipped=True),
    "pylon-color": lambda: PylonCapture("color"),
    "pylon-color-flipped": lambda: PylonCapture("color", is_flipped=True),
    "pylon-cropped": lambda: PylonCapture("cropped"),
    "pylon-cropped-flipped": lambda: PylonCapture("cropped", is_flipped=True),
    "gstreamer": GStreamerCapture,
}