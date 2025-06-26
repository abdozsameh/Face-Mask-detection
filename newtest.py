import cv2
import face_recognition
import pickle
from ultralytics import YOLO

# Load YOLOv8 model for mask detection
model = YOLO("best.pt")

# Confidence threshold for YOLO detections
CONFIDENCE_THRESHOLD = 0.9

# Load known face encodings
with open('face_encodings.pkl', 'rb') as f:
    data = pickle.load(f)
known_encodings = data['encodings']
known_names = data['names']

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

process_this_frame = 0
face_locations = []
face_names = []

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Run face recognition every 10 frames
    if process_this_frame % 10 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        for encoding in face_encodings:
            name = "Unknown"
            matches = face_recognition.compare_faces(known_encodings, encoding)
            if True in matches:
                matched_index = matches.index(True)
                name = known_names[matched_index]
            face_names.append(name)

    process_this_frame += 1

    # Scale face locations back to full resolution
    scaled_face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]

    # Run YOLO mask detection
    results = model(frame, stream=True)

    detections = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue  # Skip low confidence

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append((x1, y1, x2, y2, label, conf))

    # Remove overlapping detections (naive IoU filter)
    filtered_detections = []
    for i, (x1, y1, x2, y2, label, conf) in enumerate(detections):
        keep = True
        for xx1, yy1, xx2, yy2, _, _ in filtered_detections:
            inter_x1 = max(x1, xx1)
            inter_y1 = max(y1, yy1)
            inter_x2 = min(x2, xx2)
            inter_y2 = min(y2, yy2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (xx2 - xx1) * (yy2 - yy1)
            iou = inter_area / float(area1 + area2 - inter_area)

            if iou > 0.5:
                keep = False
                break

        if keep:
            filtered_detections.append((x1, y1, x2, y2, label, conf))

    # Draw final results with face names
    for x1, y1, x2, y2, label, conf in filtered_detections:
        name = "Unknown"
        for (top, right, bottom, left), recognized_name in zip(scaled_face_locations, face_names):
            if x1 < right and x2 > left and y1 < bottom and y2 > top:
                name = recognized_name
                break

        color = (0, 255, 0) if label == "with mask" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} - {label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face + Mask Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
