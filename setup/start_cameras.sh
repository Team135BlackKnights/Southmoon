#!/bin/bash

echo "Starting all camera instances..." | tee -a /Users/pennrobotics/Documents/GitHub/Southmoon/camera_log.txt

# Activate virtual environment
source ~/venvs/southmoon/bin/activate

# Define a function to continuously run a given instance.
# The first argument is the config file and the second is the calibration file.
run_instance() {
    local config_file="$1"
    local calib_file="$2"
    while true; do
        python /Users/pennrobotics/Documents/GitHub/Southmoon/init.py --config "$config_file" --calibration "$calib_file"
        echo "$(date): Instance ($config_file, $calib_file) failed. Restarting in 5 seconds." | tee -a /Users/pennrobotics/Documents/GitHub/Southmoon/camera_log.txt
        sleep 5
    done
}

# Start each instance in its own background thread
run_instance /Users/pennrobotics/Documents/GitHub/Southmoon/setup/config_frontCam.json /Users/pennrobotics/Documents/GitHub/Southmoon/setup/calibration_frontCam.json &
#run_instance /Users/pennrobotics/Documents/GitHub/Southmoon/setup/config_extraFrontCam.json /Users/pennrobotics/Documents/GitHub/Southmoon/setup/calibration_extraFrontCam.json &
#run_instance /Users/pennrobotics/Documents/GitHub/Southmoon/setup/config_backCam.json /Users/pennrobotics/Documents/GitHub/Southmoon/setup/calibration_backCam.json &
#run_instance /Users/pennrobotics/Documents/GitHub/Southmoon/setup/config_rightCam.json /Users/pennrobotics/Documents/GitHub/Southmoon/setup/calibration_rightCam.json &

# Wait for all background jobs to finish (which they never should unless you kill the script)
wait