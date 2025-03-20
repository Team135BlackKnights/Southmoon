#!/bin/bash
cd /Users/pennrobotics/Southmoon

echo "Starting all camera instances..." | tee -a ~/camera_log.txt

# Start 4 instances with different config/calibration files
python3 init.py --config /Users/pennrobotics/Southmoon/setup/config_frontCam.json --calibration /Users/pennrobotics/Southmoon/setup/calibration_frontCam.json &  
python3 init.py --config /Users/pennrobotics/Southmoon/setup/config_extraFrontCam.json --calibration /Users/pennrobotics/Southmoon/setup/calibration_extraFrontCam.json &  
python3 init.py --config /Users/pennrobotics/Southmoon/setup/config_backCam.json --calibration calibration_backCam.json &  
python3 init.py --config /Users/pennrobotics/Southmoon/setup/config_rightCam.json --calibration /Users/pennrobotics/Southmoon/setup/calibration_rightCam.json &  

# Wait to prevent immediate script exit
wait
