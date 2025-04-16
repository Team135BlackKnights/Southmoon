import cv2

# Open the camera (assuming index 1 is the Arducam)
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

# Check if the camera was successfully opened
if not cap.isOpened():
    print("Error: Could not open camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Verify the frame format
    if frame is not None:
        # Convert the frame from 'UYVY422' to BGR (OpenCV uses 'cv2.COLOR_YUV2BGR_UYVY' for this conversion)
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
            # Display the frame
            cv2.imshow('Live Camera Feed', frame_bgr)
        except cv2.error as e:
            print(f"Error in conversion: {e}")
            continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
