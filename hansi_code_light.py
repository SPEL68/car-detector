import logging
import cv2
import csv
import datetime
import numpy as np
from ultralytics import YOLO
from sort import *

# Lower Ultralytics logging to WARNING
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Example COCO class list to match YOLO model indices
class_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Replace with your RTSP stream URL
rtsp_url = "rtsp://Kamera02:sehvemdersmutterforbi@192.168.30.231:554/stream1"

# Load YOLOv8 model
model = YOLO("yolo11n.pt")

# We only want certain classes (COCO indices): car=2, motorcycle=3, bus=5, truck=7
vehicle_classes = [2, 3, 5, 7]

# Generate a timestamp for the CSV filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"vehicle_log_{timestamp}.csv"

# Setup CSV logging
csv_file = open(csv_filename, mode="w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["Timestamp", "VehicleCount"])


"""# Setup CSV logging
csv_filename = "vehicle_log.csv"
csv_file = open(csv_filename, mode="w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["Timestamp", "VehicleCount"])
"""

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Cannot connect to RTSP stream")
    exit()

print("RTSP stream is live. Press 'q' to exit.")

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Keep track of when each unique ID was first seen
carCount = {}
Skip_True =True

while True:
    if Skip_True: 
        Skip_True = False
        continue
   
    Skip_True = True
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    # Resize frame to reduce processing load
    frame = cv2.resize(frame, (640, 360))

    # Run YOLO on the resized frame
    results = model.predict(frame, conf=0.2, iou=0.45)

    # Gather detections (x1, y1, x2, y2, confidence) for classes of interest
    detections = np.empty((0, 5))

    if len(results) > 0:
        det_boxes = results[0].boxes
        for box in det_boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update SORT tracker with current frame’s detections
    tracked_results = tracker.update(detections)

    # Mark or count new track IDs
    for result in tracked_results:
        x1, y1, x2, y2, track_id = result
        track_id = int(track_id)
        if track_id not in carCount:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            carCount[track_id] = timestamp
            writer.writerow([timestamp, len(carCount)])

        # Draw bounding box & track ID on frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add total counted text on the frame
    cv2.putText(frame, f"Vehicles Counted: {len(carCount)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # Show live video
    cv2.imshow("Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()

print(f"Done. Logged vehicle detections in '{csv_filename}'.")