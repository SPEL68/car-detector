import cv2
import numpy as np

# Load your video


# Replace with your actual RTSP URL
rtsp_url = "rtsp://Kamera02:sehvemdersmutterforbi@192.168.30.231:554/stream1"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)


# Create a mask (example: a rectangle mask)
ret, frame = cap.read()
if not ret:
    print("Failed to read the video stream")
    cap.release()
    exit()



# Create a polygon mask with the same size as the frame
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
points = np.array([[400,0], [1900, 0], [1800, 900], [1000, 500]], np.int32)
points = points.reshape((-1, 1, 2))
cv2.fillPoly(mask, [points], 255)  # Draw a filled polygon
"""
# Ensure the mask has the same size as the frame
mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Adjust the size to match your video frame size
cv2.rectangle(mask, (400, 0), (3000, 1000), 255, -1)  # Example coordinates
"""
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Visualize the mask and the masked frame
    cv2.imshow('Mask', mask)
    cv2.imshow('Masked Frame', masked_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()