# Southmoon

Northstar is 6328's AprilTag tracking and object detection system.

Southmoon, on the other hand, is 135's, and has been built around easier development, more efficient Aruco/Apriltag detection, and more cameras.

To get a new Mac Mini set up, you will need to use Time Machine.
STEPS:
0. Get the Github installed, and cd into the directory.
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
            <string>/Users/USERNAME!!!/PathToSouthmoon/start_cameras.sh</string>
        </array>

        <key>RunAtLoad</key>
        <true/>

        <key>KeepAlive</key>
        <dict>
            <key>SuccessfulExit</key>
            <false/>
        </dict>

        <key>StandardOutPath</key>
        <string>/Users/USERNAME!!!/PathToSouthmoon/apriltag_stdout.log</string>

        <key>StandardErrorPath</key>
        <string>/Users/USERNAME!!!/PathToSouthmoon/apriltag_stderr.log</string>
    </dict>
    </plist>
6. launchctl load ~/Library/LaunchAgents/com.apriltag.multicamera.plist
7. launchctl list | grep apriltag
8. 

CHECKING:
launchctl unload ~/Library/LaunchAgents/com.apriltag.multicamera.plist
launchctl load ~/Library/LaunchAgents/com.apriltag.multicamera.plist
Use advantage scope to view the cameras, looking for print_log

Updating:
You have a few ways to update; 
1. Use a hotspot/SSH and wirelessly update the Github repo, then using 'git pull' inside the mac mini via SSH
    To do this, use ssh pennrobotics@10.1.35.200 (the default IP for a new Mac on a RoboRIO), then cd ~/PathToSouthmoon, then git pull, then unload/load the launch agent
2. Use only SSH, directly using nano text editor to change the files as you need. You may end up wanting to use 'mv source_file.txt /path/to/destination/' That can rename too (for configs)

In the event that the Mac Mini has data to UPLOAD, EXCEPT FOR MATCH FOOTAGE, do this:
    Use a hotspot/SSH and use git add . in the southmoon directory, then git commit -m "Commit Name", then git push
For MATCH footage, upload that to a match footage folder in the Team Google Drive