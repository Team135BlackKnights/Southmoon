#!/bin/bash
cd /Users/kbmarsh88/Programming/135-TSL/Southstar

echo "Starting all camera instances..." | tee -a ~/camera_log.txt

# Start 4 instances with different config/calibration files
/usr/local/bin/python3 init.py --config config_frontCam.json --calibration calibration_frontCam.json &  
/usr/local/bin/python3 init.py --config config_extraFrontCam.json --calibration calibration_extraFrontCam.json &  
/usr/local/bin/python3 init.py --config config_backCam.json --calibration calibration_backCam.json &  
/usr/local/bin/python3 init.py --config config_rightCam.json --calibration calibration_rightCam.json &  

# Wait to prevent immediate script exit
wait