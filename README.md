# Southmoon

Northstar is 6328's AprilTag tracking and object detection system.

Southmoon, on the other hand, is 135's, and has been built around easier development, more efficient Aruco/Apriltag detection, CoreML object detection, and more cameras.

This README is partly setup notes, partly "what you should understand if you are now the poor soul maintaining vision." If you are reading this during a competition, read the commands sections first. If you are reading this before build season, read all of it.

Useful outside docs:
- PhotonVision docs: https://docs.photonvision.org/en/latest/index.html
- PhotonVision robot pose estimator: https://docs.photonvision.org/en/latest/docs/programming/photonlib/robot-pose-estimator.html
- WPILib AprilTags: https://docs.wpilib.org/en/stable/docs/software/vision-processing/apriltag/apriltag-intro.html
- WPILib NetworkTables: https://docs.wpilib.org/en/stable/docs/software/networktables/networktables-intro.html
- AdvantageKit docs: https://docs.advantagekit.org/
- AdvantageScope docs: https://docs.advantagescope.org/
- Limelight docs: https://docs.limelightvision.io/docs/docs-limelight/getting-started/summary
- OpenCV solvePnP docs: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
- OpenCV camera calibration: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- OpenCV ArUco/ChArUco calibration: https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html
- Ultralytics YOLO docs: https://docs.ultralytics.com/
- CoreML / coremltools: https://apple.github.io/coremltools/docs-guides/source/unified-conversion-api.html
- Roboflow Universe: https://universe.roboflow.com/
- macOS Remote Login / SSH: https://support.apple.com/guide/mac-help/allow-a-remote-computer-to-access-your-mac-mchlp1066/mac
- launchctl man page: https://keith.github.io/xcode-man-pages/launchctl.1.html

## What Southmoon Actually Does

There are two sides:

1. The Mac side runs `start_cameras.sh`, which starts one `init.py` process per camera.
2. The robot side publishes config over NetworkTables and reads the output packets.

The Mac does the expensive stuff:
- camera capture
- AprilTag / ArUco style detection
- CoreML object detection
- optional video recording
- printing logs back to NetworkTables

The robot does the trust stuff:
- "is this pose sane?"
- "what standard deviation should this vision measurement use?"
- "which object detection should commands actually aim at?"
- "is this camera connected or just silently dead?"

If you change packet formats, config names, class IDs, camera names, or camera order, you need to check BOTH sides. Do not assume Python and Java magically agree.

## PhotonVision vs Our Custom Stuff

PhotonVision is the normal FRC tool. It is good, documented, and everyone uses it. For normal AprilTags it can do camera calibration, tag detection, and robot pose estimation.

We still support PhotonVision-style inputs robot-side. The robot code has `VisionIO` so cameras can provide already-parsed `TargetObservation` and `PoseObservation` objects.

Southmoon exists because we needed things PhotonVision was not giving us in the exact way we wanted:
- Mac-side CoreML object models
- more custom camera control
- multiple cameras with our own config flow
- our own object `tx/ty` packet format
- our own video recording logic
- custom filtering and clustering robot-side

So, PhotonVision is not "bad." We just outgrew the easy path for this robot.

## AprilTags, ArUco, And The "Known Square" Trick

AprilTags and ArUco markers are both fiducials. A fiducial is just a target where the computer can say:

"I know this square's ID, I know how big it is, and I know where its corners are supposed to be in 3D."

The camera sees four corners in pixels. The code knows the real 3D corner layout. That lets solvePnP estimate the camera pose.

Important pieces that MUST agree:
- tag ID
- tag physical width
- field layout JSON
- camera calibration
- robot-to-camera mounting pose
- image resolution

If any of those are wrong, the camera will still give you a pose. It will just be wrong, which is worse than crashing because it looks real.

## solvePnP, Camera Intrinsics, And Why You Recalibrate

solvePnP means "solve Perspective-n-Point." In normal human words:

"Given some known 3D points, and where those points appeared in the camera image, where is the camera in 3D?"

For an AprilTag:
- the known 3D points are the tag corners
- the measured 2D points are the pixel corners
- the answer is camera translation + camera rotation

The rough 3D process:

1. Detect the tag corners in the image.
2. Build the real-world 3D corner points from the tag size.
3. Use the camera matrix to turn pixels into rays.
4. Use distortion coefficients to undo lens bending.
5. Run solvePnP to find `rvec` and `tvec`.
6. Convert OpenCV coordinates into WPILib coordinates.
7. Use the known field tag pose to get field-to-camera.
8. Use robot-to-camera transform to get field-to-robot.

The camera matrix is NOT just a random math thing. It contains:
- focal length in x pixels
- focal length in y pixels
- optical center x
- optical center y

The distortion coefficients describe how the lens bends straight lines.

This is why you cannot just change a lens. A new lens changes focal length and distortion.

This is why you cannot just change resolution. The camera matrix is in pixel units, so a 1600x1304 calibration is not automatically valid at 1280x720. Maybe some scaling math could be done, but do not assume that at comp. Recalibrate.

Recalibrate when:
- the camera changes
- the lens changes
- resolution changes
- crop changes
- focus changes a lot
- the mount gets hit and the pose is suspicious

Southmoon loads this from `calibration.json`, `calibration2.json`, etc. Robot-side `VisionConstants.cameras` is the robot-to-camera transform. Calibration is "how the lens sees." The robot pose constant is "where the camera is mounted."

## Limelight Object Pipelines

Limelight is still its own thing. Do not confuse it with Southmoon.

Limelight object pipelines are good for quick targeting, especially when you can tune by color/thresholds or use Limelight's own neural pipeline tools. Southmoon object detection is our Mac/CoreML path.

Things to check on a Limelight:
- pipeline number
- exposure
- thresholding
- target color
- `tx`, `ty`, `ta`, and pipeline latency
- whether NetworkTables has the right Limelight name

Useful Limelight docs:
- General docs: https://docs.limelightvision.io/docs/docs-limelight/getting-started/summary
- Color thresholding: https://docs.limelightvision.io/docs/docs-limelight/pipeline-retro/color-filtering
- Neural networks: https://docs.limelightvision.io/docs/docs-limelight/pipeline-neural/getting-started-with-neural-networks

## Getting Object Models From Roboverse / Roboflow Universe

I keep saying Roboverse sometimes. The real thing people usually mean is Roboflow Universe.

Go search for FRC datasets:
- `FRC`
- the game year
- the game piece name
- `robot bumper`
- `FRC note`
- `FRC algae`
- `FRC coral`
- whatever object you are actually trying to detect

Do NOT just download a random model and expect it to work.

Check:
- are the labels actually good?
- are there enough real field images?
- are there weird labels on background stuff?
- does the dataset include bad lighting?
- does it include motion blur?
- does it include partial objects?
- does the class order match what robot code expects?

Class order matters a lot. If the model says class `0 = FUEL`, then Java's `AITargets.FUEL.ordinal()` needs to be `0`. If you add classes, update Java too.

## Training Custom Object Models In 3 Steps

The repo has three scripts for the simple model workflow:

1. `setup/train.py`
2. `setup/export_coreml.py`
3. `setup/testPT.py`

YES, it is `export_coreml.py`. If you type `expore_coreml.py`, that is just a typo and it will not work.

### Step 1: Train

Usually this should happen on a PC with a real GPU. It can run CPU, but you will suffer.

Example:

```bash
python3 setup/train.py \
  --data /path/to/dataset/data.yaml \
  --model yolo26n.pt \
  --imgsz 640 \
  --epochs 150 \
  --batch 32 \
  --name fuel_model
```

Output you care about:

```bash
runs_detect_coreml/fuel_model/weights/best.pt
```

That `.pt` is the trained PyTorch model.

### Step 2: Export To CoreML On macOS

The Mac needs a CoreML package/model.

Example:

```bash
python3 setup/export_coreml.py \
  --weights runs_detect_coreml/fuel_model/weights/best.pt \
  --imgsz 640 \
  --out /Users/pennrobotics/Documents/GitHub/Southmoon/Setup/best.mlpackage
```

The config points to the model here:

```json
"obj_detect_model": "/Users/pennrobotics/Documents/GitHub/Southmoon/Setup/best.mlpackage"
```

It can also be an `.mlmodel` depending on what you exported, but `.mlpackage` is what the script is expecting now.

### Step 3: Test The `.pt`

Test before you blame the robot.

Webcam:

```bash
python3 setup/testPT.py \
  --weights runs_detect_coreml/fuel_model/weights/best.pt \
  --source 0 \
  --show
```

Image:

```bash
python3 setup/testPT.py \
  --weights runs_detect_coreml/fuel_model/weights/best.pt \
  --source /path/to/test_image.jpg \
  --save /path/to/output.jpg
```

If the `.pt` is bad, the CoreML export will also be bad. Exporting does not magically fix training.

## Color vs Monochrome Models

Color models learn color and shape. They are good when the object has a very obvious color and the lighting is stable.

Color model problems:
- field lighting changes
- exposure changes
- LED reflections
- shadows
- object gets dirty
- the background has the same color

Monochrome models learn more shape/texture. They are usually less dependent on lighting color, but they need better data because you removed a very useful clue.

For both:
- train with real camera images if possible
- include far/near examples
- include blurry examples
- include partly blocked objects
- include "no target" backgrounds
- keep class order stable

## NetworkTables 4

NetworkTables is how robot code and the Mac talk.

Southmoon uses paths like:

```text
/<deviceId>/config/*
/<deviceId>/output/*
```

Examples:

```text
/IntakeCam/config/camera_id
/IntakeCam/config/print_log/log
/IntakeCam/output/observations
/IntakeCam/output/objdetect_txy
/IntakeCam/output/fps_apriltags
/IntakeCam/output/fps_objdetect
```

Use AdvantageScope to view this. OutlineViewer also works. If you cannot see the camera in NT, either the Mac is not connected, the device ID is wrong, the robot IP is wrong, or the process is dead.

The `print_log` path is especially useful because `init.py` wraps `print(...)` and publishes it to NT:

```text
/<cameraName>/config/print_log/log
```

So for the intake camera:

```text
/IntakeCam/config/print_log/log
```

## AdvantageKit Logging

Inputs are raw truth. Outputs are robot-code conclusions.

Examples of inputs:
- camera connected
- raw NT packets
- target observations
- FPS values

Examples of outputs:
- accepted robot pose
- rejected robot pose
- calculated standard deviation
- selected object target
- dynamic obstacle box

Do not record an output just because you feel like it might maybe be interesting. If it is not needed, do not log it.

Why? Because Replay exists. If the input is already logged, you can add an output later, rerun replay, and produce that value after the fact. Logs that contain every random thought are harder to debug.

Log outputs when:
- it explains a decision
- it is hard to reconstruct
- it is useful in AdvantageScope
- it helps with a real failure mode

## Robot-Side Vision Filtering

The robot should not blindly trust vision. Vision is helpful until it is confidently wrong.

Robot-side filters include:
- no tags = reject
- high ambiguity on one tag = reject
- impossible Z height = reject
- yaw too far from gyro = reject
- pose off the field = reject
- low object confidence = reject
- old object frame = ignore
- duplicate detections = cluster

Standard deviation is how we say "I believe this, but only this much." Farther tags get weaker. More tags get stronger. Ambiguous single-tag solves do not get trusted for rotation.

Object detections are clustered because a model may draw multiple boxes around the same real object. The best cluster is chosen by confidence, distance, count, and how centered it is.

## Basic SSH

SSH means "open a terminal on the Mac from another computer."

Base command for the Mac on robot DHCP:

```bash
ssh pennrobotics@10.1.35.200
```

Password:

```text
0135
```

If that IP does not work, check the network, hotspot, DHCP, or what address the Mac actually got.

## Basic Unix Commands You Actually Need

Show where you are:

```bash
pwd
```

List files:

```bash
ls
```

List files with details and hidden files:

```bash
ls -la
```

Go to Southmoon:

```bash
cd ~/Documents/GitHub/Southmoon
```

Go up one folder:

```bash
cd ..
```

Print a file:

```bash
cat config.json
```

Scroll through a file:

```bash
less start_cameras.sh
```

Edit a file:

```bash
nano start_cameras.sh
```

Copy a file:

```bash
cp exampleConfig.json config4.json
```

Move or rename a file:

```bash
mv config4.json old_config4.json
```

Delete a file:

```bash
rm old_config4.json
```

Be careful with `rm`. It does not ask if you meant it.

Make a folder:

```bash
mkdir videos_backup
```

Search for text:

```bash
grep -R "IntakeCam" .
```

If `rg` exists, use it because it is faster:

```bash
rg "IntakeCam"
```

Show Python processes:

```bash
ps aux | grep python
```

Watch a log live:

```bash
tail -f ~/Library/Logs/Southmoon/camera_stderr.log
```

Copy a file from your laptop to the Mac:

```bash
scp best.mlpackage pennrobotics@10.1.35.200:/Users/pennrobotics/Documents/GitHub/Southmoon/Setup/
```

Reboot:

```bash
sudo reboot
```

## Updating The Mac Wirelessly

Connect:

```bash
ssh pennrobotics@10.1.35.200
```

Password:

```text
0135
```

Go to Southmoon:

```bash
cd ~/Documents/GitHub/Southmoon
```

Update:

```bash
git pull
```

Restart Southmoon. The plist in this repo is:

```text
com.southmoon.cameras.plist
```

It runs:

```text
/Users/pennrobotics/Documents/GitHub/Southmoon/start_cameras.sh
```

Try this first:

```bash
launchctl kickstart -k gui/$(id -u)/com.southmoon.cameras
```

If you need to fully unload/reload:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.southmoon.cameras.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.southmoon.cameras.plist
```

Check it:

```bash
launchctl print gui/$(id -u)/com.southmoon.cameras
```

If the plist is not installed yet:

```bash
mkdir -p ~/Library/LaunchAgents
cp com.southmoon.cameras.plist ~/Library/LaunchAgents/com.southmoon.cameras.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.southmoon.cameras.plist
```

Old `launchctl load` / `launchctl unload` may still work, but `bootstrap` / `bootout` is the newer macOS way.

If everything is cooked:

```bash
sudo reboot
```

## New Mac Mini Setup

To get a new Mac Mini set up, you will probably use Time Machine first.

After that, check:

0. Get GitHub installed / cloned, and `cd` into Southmoon.
1. Make sure `start_cameras.sh` is executable.
2. Make sure the plist is in `~/Library/LaunchAgents`.
3. Make sure the plist points at the right username and Southmoon path.
4. Make sure the venv exists at `/Users/pennrobotics/venvs/southmoon`.
5. Make sure the logs folder exists.

Commands:

```bash
cd ~/Documents/GitHub/Southmoon
chmod +x start_cameras.sh
mkdir -p ~/Library/LaunchAgents
mkdir -p ~/Library/Logs/Southmoon
cp com.southmoon.cameras.plist ~/Library/LaunchAgents/com.southmoon.cameras.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.southmoon.cameras.plist
launchctl print gui/$(id -u)/com.southmoon.cameras
```

Use AdvantageScope to view the cameras, looking for `print_log`, FPS, and output packets.

## Checking Logs

NetworkTables:

```text
/IntakeCam/config/print_log/log
/BackRightCam/config/print_log/log
/BackLeftCam/config/print_log/log
```

Mac log files:

```bash
tail -f ~/Library/Logs/Southmoon/camera_log.txt
tail -f ~/Library/Logs/Southmoon/camera_stdout.log
tail -f ~/Library/Logs/Southmoon/camera_stderr.log
```

If you see "camera not found", check:
- camera USB ID
- camera location
- cable
- config file
- `VisionConstants.cameras`
- whether the camera is already opened by another process

## Adding New Cameras

New camera setup touches multiple places. Do not only change one file and call it done.

Mac side:
- make a new config file, probably copied from `exampleConfig.json`
- make a new calibration file
- add a line in `start_cameras.sh`

Example:

```bash
run_camera "config4.json" "calibration4.json" "ExtraCam" &
```

Robot side:
- add/update the camera in `VisionConstants.cameras`
- make sure the device ID matches the config
- make sure camera order matches how robot code expects it
- set the robot-to-camera transform correctly

Then restart:

```bash
launchctl kickstart -k gui/$(id -u)/com.southmoon.cameras
```

## Uploading Changes From The Mac

In the event that the Mac Mini has data to UPLOAD, EXCEPT FOR MATCH FOOTAGE:

```bash
cd ~/Documents/GitHub/Southmoon
git status
git add .
git commit -m "Commit Name"
git push
```

For MATCH footage, upload that to a match footage folder in the Team Google Drive. Do not put giant match videos in git.

