#!/bin/zsh

echo "Starting all Southmoon cameras at $(date)" >> /Users/pennrobotics/Library/Logs/Southmoon/camera_log.txt

# Activate virtual environment
source /Users/pennrobotics/venvs/southmoon/bin/activate

# Navigate to project
cd /Users/pennrobotics/Documents/GitHub/Southmoon

# Function to run a camera instance with auto-restart
run_camera() {
    local config_file="$1"
    local calib_file="$2"
    local camera_name="$3"
   
    while true; do
        echo "$(date): Starting $camera_name" >> /Users/pennrobotics/Library/Logs/Southmoon/camera_log.txt
        python init.py --config "$config_file" --calibration "$calib_file" 2>&1 | while IFS= read -r line; do
            echo "[$camera_name] $line"
        done
        echo "$(date): $camera_name exited. Restarting in 5 seconds..." >> /Users/pennrobotics/Library/Logs/Southmoon/camera_log.txt
        sleep 5
    done
}

# Start each camera in background
# Uncomment the cameras you want to use

#run_camera "config.json" "calibration.json" "DefaultCam" &

# If you have multiple cameras, add them here:
 run_camera "config.json" "calibration.json" "FrontCam" &
 run_camera "config2.json" "calibration.json" "BackCam" &
# run_camera "setup/config_rightCam.json" "setup/calibration_rightCam.json" "RightCam" &
# run_camera "setup/config_extraFrontCam.json" "setup/calibration_extraFrontCam.json" "ExtraFrontCam" &

# Wait for all background processes
wait
