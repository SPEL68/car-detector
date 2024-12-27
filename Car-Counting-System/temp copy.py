from ultralytics import YOLO
import cv2
import numpy as np

from sort import *
model = YOLO('yolov8n.pt')

class_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Replace with your actual RTSP URL
rtsp_url = "rtsp://Kamera02:sehvemdersmutterforbi@192.168.30.231:554/stream2"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Create a blank image (mask) with the same dimensions as your 2K video stream
height, width = 1296, 2304  # 2K resolution for Tapo C310
mask = np.zeros((height, width), dtype=np.uint8)

# Define the polygon coordinates (example coordinates, adjust as needed)
polygon = np.array([[250, 0], [2300, 1200], [2300, 250], [1100, 0]], np.int32)
polygon = polygon.reshape((-1, 1, 2))

# Fill the polygon on the mask
cv2.fillPoly(mask, [polygon], 255)

# Save or use the mask as needed
cv2.imwrite('polygon_mask.png', mask)  # Optional: Save the mask if needed

# Read the mask
mask = cv2.imread('polygon_mask.png', cv2.IMREAD_GRAYSCALE)
mask_for_detection = cv2.imread('polygon_mask.png')
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

overlay_color = (0, 255, 0)  # Green color
alpha = 0.5  # Transparency factor (0: fully transparent, 1: fully opaque)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [350, 250, 550, 30]
#limits = [160, 400, 530, 400]
totalCount = []

while True:
    success, image = cap.read()
    if not success:
        break

    resize = cv2.resize(image, (700, 500))

    # Resize the mask to match the resized image dimensions
    resized_mask = cv2.resize(mask, (700, 500), interpolation=cv2.INTER_NEAREST)
    resized_mask_for_detection = cv2.resize(mask_for_detection, (700, 500), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros_like(resize)
    color_mask[resized_mask == 255] = overlay_color

    mask_added = cv2.addWeighted(resize, 1, color_mask, alpha, 0)

    imgRegion = cv2.bitwise_and(resize, resized_mask_for_detection)

    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            currentClass = class_list[int(box.cls[0])]
            currentConf = box.conf[0]
            if currentClass == "car" and currentConf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, currentConf])
                detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)
    cv2.line(mask_added, (350, 250), (550, 30), (0, 0, 255), 5)
    #cv2.line(mask_added, (160, 400), (530, 400), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        w, h = x2 - x1, y2 - y1
        cx, cy = int(x1 + w // 2), int(y1 + h // 2)
        cv2.rectangle(mask_added, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.putText(mask_added, str(int(Id)), (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2, cv2.LINE_AA)
        cv2.circle(mask_added, (cx, cy), 5, (0, 255, 0), -1)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if totalCount.count(Id) == 0:
                totalCount.append(Id)

        cv2.putText(mask_added, "Cars Count: " + str(len(totalCount)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2, cv2.LINE_AA, False)

    cv2.imshow("Image", mask_added)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
#cv2.destroyAllWindows()