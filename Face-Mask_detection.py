import cv2  
import face_recognition 
import pickle  # 3shan n-load el saved encodings mn file
from ultralytics import YOLO  

# Load YOLOv8 model for mask detection
model = YOLO("best.pt")  # load el model el matdrab 3la mask detection

# Confidence threshold for YOLO detections
CONFIDENCE_THRESHOLD = 0.9  # el threshold el 2al mnha nskip el detection

# Load known face encodings
with open('face_encodings.pkl', 'rb') as f:  # n-open el pickle file el feh el data
    data = pickle.load(f)
known_encodings = data['encodings']  # list b feha features l kol wesh
known_names = data['names']  # el asamy l mortabta bel features dol

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # n7aded el width bta3 el frame
cap.set(4, 720)  # n7aded el height bta3 el frame

process_this_frame = 0   # counter 3shan n2all el processing to reduce lagging
face_locations = []  # list feha makan el wesh el mt3araf 3leh
face_names = []  # list feha asamy el wesh el mt3araf 3leh

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()  # n-read frame mn el camera
    if not ret:  # lw mafesh frame ytl3 break
        break

    frame = cv2.flip(frame, 1)  # n3ml mirror l image 3shan tb2a zay el wa2e3y

    # Run face recognition kol 10 frames bs
    if process_this_frame % 10 == 0: 
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # n2ll el size l tasre3
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # YOLO w face lib b7tago RGB

        face_locations = face_recognition.face_locations(rgb_small)  # n-detect makan el wesh
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)  # n-akhod encodings

        face_names = []  # n-clear el list
        for encoding in face_encodings:
            name = "Unknown" 
            matches = face_recognition.compare_faces(known_encodings, encoding)  # nshof lw feh match
            if True in matches:
                matched_index = matches.index(True)  # ngeb el index bta3 el match
                name = known_names[matched_index]  # ngeb el esm
            face_names.append(name)  # nzwd el esm l list

    process_this_frame += 1  # nzwd el counter

    # Scale face locations l original size
    scaled_face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]  # nrg3 el coordinates l size el asli

    # Run YOLO mask detection
    results = model(frame, stream=True)  # n-detect mask b YOLO
    detections = []  # list feha kol el detections

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])  # n-akod el confidence
            if conf < CONFIDENCE_THRESHOLD:  # lw el confidence 2l mn el threshold skip
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # el coordinates bta3 el box
            cls = int(box.cls[0])  # class index
            label = model.names[cls]  # esm el class: with mask / without mask
            detections.append((x1, y1, x2, y2, label, conf))  # nzwd el detection l list

    # Filter overlapping detections using IoU (Intersection over Union)
    filtered_detections = []  # list lel detections bdon overlap
    for x1, y1, x2, y2, label, conf in detections:
        keep = True
        for xx1, yy1, xx2, yy2, _, _ in filtered_detections:
            inter_x1 = max(x1, xx1)  # el bdaya el mshtrka
            inter_y1 = max(y1, yy1)
            inter_x2 = min(x2, xx2)  # el nihaya el mshtrka
            inter_y2 = min(y2, yy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)  # masahat el intersect
            area1 = (x2 - x1) * (y2 - y1)  # masahat el box el awl
            area2 = (xx2 - xx1) * (yy2 - yy1)  # masahat el box el tany
            iou = inter_area / float(area1 + area2 - inter_area)  # el ratio el mshtrka
            if iou > 0.5:  # lw overlap kbeer skip
                keep = False
                break
        if keep:
            filtered_detections.append((x1, y1, x2, y2, label, conf))  # nzwd el box el msh overlapped

    # Draw final results with proper label
    for x1, y1, x2, y2, label, conf in filtered_detections:
        name = "Unknown"
        for (top, right, bottom, left), recognized_name in zip(scaled_face_locations, face_names):
            if x1 < right and x2 > left and y1 < bottom and y2 > top:  # lw el box el YOLO ytmasha m3 el face box
                name = recognized_name  # esm el sh5s el mt3araf 3leh
                break

        if name == "Unknown":
            text = f"A stranger is {label}"  # lw msh ma3rof
        else:
            text = f"{name} is {label}"  # lw ma3rof

        color = (0, 255, 0) if label == "with mask" else (0, 0, 255)  # akhdar lw 3leh mask / ahmar lw la2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # nrsom box 3la el face
        cv2.putText(frame, text, (x1, y1 - 10),  # nktb el esm w status fo2 el box
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face + Mask Recognition", frame)  # n3rd el result
    if cv2.waitKey(1) & 0xFF == ord('q'):  # lw dost 'q' n-exit
        break

cap.release()  # nfsl el camera
cv2.destroyAllWindows()  # n2fl el windows el mfto7a
