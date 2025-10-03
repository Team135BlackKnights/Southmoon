#!/usr/bin/env python3
"""
Test script for USB camera capture with GStreamer.
This script helps verify camera detection and exposure control.
"""

import cv2
import time
import sys

def test_avfvideosrc():
    """Test avfvideosrc with different device indices."""
    print("Testing AVFoundation video sources...")
    
    found_cameras = []
    
    for idx in range(10):
        print(f"\nTesting device index {idx}...")
        
        # Try different formats
        formats = [
            ("UYVY", f"avfvideosrc device-index={idx} ! video/x-raw,format=UYVY,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"),
            ("YUY2", f"avfvideosrc device-index={idx} ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"),
            ("NV12", f"avfvideosrc device-index={idx} ! video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"),
        ]
        
        for format_name, pipeline in formats:
            try:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"  ✓ Device {idx} works with format {format_name}")
                        print(f"    Frame shape: {frame.shape}")
                        found_cameras.append({
                            'index': idx,
                            'format': format_name,
                            'pipeline': pipeline
                        })
                        cap.release()
                        break  # Found working format, move to next device
                    cap.release()
            except Exception as e:
                pass
    
    print(f"\n{'='*60}")
    print(f"Found {len(found_cameras)} working camera(s):")
    for cam in found_cameras:
        print(f"  Device {cam['index']}: {cam['format']}")
    print(f"{'='*60}\n")
    
    return found_cameras


def test_camera_capture(device_index=0, width=1280, height=720, fps=60):
    """Test camera capture with exposure control simulation."""
    print(f"\nTesting camera {device_index} at {width}x{height}@{fps}fps...")
    
    # Build pipeline with UYVY format for best control
    pipeline = (
        f"avfvideosrc device-index={device_index} "
        f"capture-screen=false "
        f"capture-screen-cursor=false "
        f"! video/x-raw,format=UYVY,width={width},height={height},framerate={fps}/1 "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink drop=1 max-buffers=2"
    )
    
    print(f"Pipeline: {pipeline}")
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("✗ Failed to open camera")
        return False
    
    print("✓ Camera opened successfully")
    
    # Warm up
    print("Warming up camera...")
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"  Frame {i+1}: {frame.shape}")
        time.sleep(0.05)
    
    # Test frame capture
    print("\nCapturing test frames...")
    frame_count = 0
    start_time = time.time()
    
    for i in range(60):  # Capture 60 frames
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frame_count += 1
            
            # Display frame info every 10 frames
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Frame {frame_count}: {frame.shape}, FPS: {actual_fps:.1f}")
        else:
            print(f"  ✗ Failed to capture frame {i+1}")
    
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Captured {frame_count}/60 frames")
    print(f"Average FPS: {actual_fps:.1f}")
    print(f"{'='*60}\n")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return frame_count > 0


def main():
    print("="*60)
    print("USB Camera Test with GStreamer/AVFoundation")
    print("="*60)
    
    # First, find all available cameras
    cameras = test_avfvideosrc()
    
    if not cameras:
        print("\n✗ No cameras found!")
        print("\nTroubleshooting:")
        print("1. Make sure a USB camera is connected")
        print("2. Check that GStreamer is installed: brew install gstreamer gst-plugins-base gst-plugins-good")
        print("3. Test with: gst-launch-1.0 avfvideosrc device-index=0 ! autovideosink")
        return 1
    
    # Test the first camera found
    if cameras:
        print(f"\nTesting first camera (device {cameras[0]['index']})...")
        
        # Test at different resolutions
        test_configs = [
            (640, 480, 30),
            (1280, 720, 60),
            (1920, 1080, 30),
        ]
        
        for width, height, fps in test_configs:
            print(f"\n{'='*60}")
            print(f"Testing {width}x{height}@{fps}fps")
            print(f"{'='*60}")
            
            if not test_camera_capture(cameras[0]['index'], width, height, fps):
                print(f"✗ Failed at {width}x{height}@{fps}fps")
            else:
                print(f"✓ Success at {width}x{height}@{fps}fps")
            
            time.sleep(1)
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
