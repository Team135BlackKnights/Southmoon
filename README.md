# Southmoon

Northstar is 6328's AprilTag tracking and object detection system.

Southmoon, on the other hand, is 135's, and has been built around easier development, more efficient Aruco/Apriltag detection, and more cameras.

To get a new Mac Mini set up, you will need to use Time Machine.

STEPS:
1. nano ~/start_cameras.sh
2. grab new file location
3. chmod +x ~/start_cameras.sh
4. 
    mkdir -p ~/Library/LaunchAgents
    nano ~/Library/LaunchAgents/com.apriltag.multicamera.plist
5. 
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>Label</key>
        <string>com.apriltag.multicamera</string>

        <key>ProgramArguments</key>
        <array>
            <string>/Users/USERNAME!!!/start_cameras.sh</string>
        </array>

        <key>RunAtLoad</key>
        <true/>

        <key>KeepAlive</key>
        <dict>
            <key>SuccessfulExit</key>
            <false/>
        </dict>

        <key>StandardOutPath</key>
        <string>/Users/USERNAME!!!/apriltag_stdout.log</string>

        <key>StandardErrorPath</key>
        <string>/Users/USERNAME!!!/apriltag_stderr.log</string>
    </dict>
    </plist>
6. launchctl load ~/Library/LaunchAgents/com.apriltag.multicamera.plist
7. launchctl list | grep apriltag
8. 

CHECKING:
cat ~/apriltag_stdout.log
cat ~/apriltag_stderr.log

launchctl unload ~/Library/LaunchAgents/com.apriltag.multicamera.plist
launchctl load ~/Library/LaunchAgents/com.apriltag.multicamera.plist

