import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")  

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, stream=True)

    detections = []

    # Gather boxes that pass threshold
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue  # skip low-confidence detections

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            detections.append((x1, y1, x2, y2, label, conf))

    # Optional: Remove overlapping labels (naive method by comparing IoU)
    filtered_detections = []
    for i, (x1, y1, x2, y2, label, conf) in enumerate(detections):
        keep = True
        for j, (xx1, yy1, xx2, yy2, _, _) in enumerate(filtered_detections):
            # Calculate Intersection over Union (IoU)
            inter_x1 = max(x1, xx1)
            inter_y1 = max(y1, yy1)
            inter_x2 = min(x2, xx2)
            inter_y2 = min(y2, yy2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (xx2 - xx1) * (yy2 - yy1)
            iou = inter_area / float(area1 + area2 - inter_area)

            if iou > 0.5:  # if overlapping significantly
                keep = False
                break

        if keep:
            filtered_detections.append((x1, y1, x2, y2, label, conf))

    # Draw final boxes
    for x1, y1, x2, y2, label, conf in filtered_detections:
        color = (0, 255, 0) if label == "with mask" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("YOLOv8 Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
