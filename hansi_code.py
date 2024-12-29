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
# Make sure indices 2,3,5,7 match 'car','motorcycle','bus','truck'
class_list = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Replace with your RTSP stream URL
rtsp_url = "rtsp://Kamera02:sehvemdersmutterforbi@192.168.30.231:554/stream1"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# We only want certain classes (COCO indices): car=2, motorcycle=3, bus=5, truck=7
vehicle_classes = [2, 3, 5, 7]

# Setup CSV logging
csv_filename = "vehicle_log.csv"
csv_file = open(csv_filename, mode="w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["Timestamp", "VehicleCount"])

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

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    # Get original dimensions
    height, width, _ = frame.shape

    # Calculate the slice boundaries
    top_cut = int(0.1 * height)
    left_cut = int(0.25 * width)  # skip left 20%

    # Perform the cropping
    cropped_frame = frame[top_cut:height, left_cut:width]

    # Run YOLO on the cropped image
    results = model.predict(cropped_frame, conf=0.25, iou=0.45)

    # 2) Gather detections (x1, y1, x2, y2, confidence) for classes of interest
    detections = np.empty((0, 5))

    if len(results) > 0:
        # results is a list of `ultralytics.yolo.engine.results.Results`,
        # typically of length 1 for a single image
        det_boxes = results[0].boxes
        for box in det_boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in vehicle_classes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # 3) Update SORT tracker with current frame’s detections
    tracked_results = tracker.update(detections)

    # 4) Mark or count new track IDs
    for result in tracked_results:
        x1, y1, x2, y2, track_id = result
        track_id = int(track_id)  # ensure int
        if track_id not in carCount:
            # First time we see this ID; record timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            carCount[track_id] = timestamp
            # Optionally, log to CSV if you want per-new-vehicle
            writer.writerow([timestamp, len(carCount)])

        # Draw bounding box & track ID on frame
        cv2.rectangle(
            cropped_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
        )
        cv2.putText(
            cropped_frame,
            f"ID {track_id}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # 5) Add total counted text on the frame
    cv2.putText(
        cropped_frame,
        f"Vehicles Counted: {len(carCount)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 6) Show live video
    cv2.imshow("Vehicle Tracking", cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()

print(f"Done. Logged vehicle detections in '{csv_filename}'.")
