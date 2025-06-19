import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from sort import Sort

# ---------------------------- Config & Constants ----------------------------
VIDEO_PATH = "../videos/cars.mp4"
YOLO_MODEL_PATH = "../Yolo-Weights/yolov8s.pt"
MASK_PATH = "mask.png"
TARGET_CLASSES = {"car", "truck", "bus", "motorbike"}
CONFIDENCE_THRESHOLD = 0.3
LINE_POSITION = [403, 297, 673, 297]  # x1, y1, x2, y2

# ---------------------------- Initialization ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(YOLO_MODEL_PATH)
mask = cv2.imread(MASK_PATH)
tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)
classNames = model.names

totalCarCount = 0
countedIDs = set()

# ---------------------------- Helper Functions ----------------------------
def filter_detections(results, class_names):
    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            if class_name in TARGET_CLASSES and conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2, conf, class_name))
    return detections

def draw_detections(img, detections):
    det_array = np.empty((0, 5))
    for x1, y1, x2, y2, conf, class_name in detections:
        cvzone.putTextRect(img, f'{class_name}::{conf:.2f}', (x1, max(20, y1 - 20)), scale=1, thickness=2, offset=3, colorR=(125, 12, 51))
        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=7)
        det_array = np.vstack((det_array, [x1, y1, x2, y2, conf]))
    return det_array

def is_center_crossing_line(cx, cy, line):
    x1, y1, x2, y2 = line
    if y1 == y2:  # horizontal line
        return abs(cy - y1) <= 20  # small tolerance
    return False

def track_and_count(img, detections_array, line):
    global totalCarCount
    results = tracker.update(detections_array)

    for x1, y1, x2, y2, track_id in results.astype(int):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if track_id not in countedIDs and is_center_crossing_line(cx, cy, line):
            countedIDs.add(track_id)
            totalCarCount += 1

        cvzone.putTextRect(img, f'ID: {track_id}', (x1, y1 - 10), scale=1, thickness=1, colorR=(0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

# ---------------------------- Main Loop ----------------------------
while cap.isOpened():
    key = cv2.waitKey(1)

    success, frame = cap.read()
    if not success:
        break


    # Preprocess and inference
    masked_frame = cv2.bitwise_and(frame, mask)
    results = model(masked_frame, stream=True)

    # Filter and draw
    detections = filter_detections(results, classNames)
    detection_array = draw_detections(frame, detections)

    # Tracking and counting
    track_and_count(frame, detection_array, LINE_POSITION)

    # Draw line and total count
    cv2.line(frame, (LINE_POSITION[0], LINE_POSITION[1]), (LINE_POSITION[2], LINE_POSITION[3]), (0, 0, 255), 5)
    cvzone.putTextRect(frame, f'Total Cars: {totalCarCount}', (50, 50), scale=2, thickness=2, colorR=(0, 0, 255))

    # Show frame
    cv2.imshow("YOLOv8 Vehicle Tracker", frame)

# ---------------------------- Cleanup ----------------------------
cap.release()
cv2.destroyAllWindows()
