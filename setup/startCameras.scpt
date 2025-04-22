-- Save as a .scpt file in Script Editor
set logDir to "/Users/pennrobotics/Documents/GitHub/Southmoon/"
set stdoutLog to logDir & "apriltag_stdout.log"
set stderrLog to logDir & "apriltag_stderr.log"

-- Full path to the shell script
set shellScript to "/usr/local/bin/start_cameras.sh"

-- Build and execute the command
set fullCommand to "/bin/bash -c \"" & shellScript & " >> " & stdoutLog & " 2>> " & stderrLog & "\""

-- Run the command in the background
do shell script fullCommand