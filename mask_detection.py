import cv2  # lib l webcam w image processing 
import face_recognition  
import pickle  # 3shan nload el face data mn file  
from ultralytics import YOLO  # YOLOv8 model l object detection (mask)

# Load YOLOv8 mask detection model
model = YOLO("best.pt")  # b-load model YOLO l mask detection  

# Load face encodings
with open('face_encodings.pkl', 'rb') as f:  # load el encodings mn el file 
    data = pickle.load(f)

known_encodings = data['encodings']  # el features bta3t known faces
known_names = data['names']  # el asamy el mortabta bel encodings 

# Start webcam
video = cv2.VideoCapture(0)
video.set(3, 1280)  # width 1280  
video.set(4, 720)   # height 720

process_this_frame = 0  # counter 3shan n2all el processing to reduce lagging
face_locations = []
face_names = []

print("Starting webcam. Press 'q' to quit.") 

while True:
    ret, frame = video.read()  # b-yread frame mn webcam w ret do heya return w di boolean value to prevent the program clash if the camera didn't recieve a frame
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view 3shan tb2a zay el wa2e3y  

    if process_this_frame % 15 == 0:  # process kol 10 frames to reduce the lag
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # t2leel el size l tasre3 el detection
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # b-transform l RGB 3shan el library bt7tag kda

        face_locations = face_recognition.face_locations(rgb_small)  # detect the faces
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)  # get encodings

        face_names = []
        mask_status = []

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            name = "Unknown"
            matches = face_recognition.compare_faces(known_encodings, face_encoding)  # check lw feh match, di btkon array feh [true, false, ...] fr example
            if True in matches:
                name = known_names[matches.index(True)]  # get el index beta3t true to match it with the name
            face_names.append(name)

            # Scale coordinates back to original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Crop face and predict mask
            face_roi = frame[top:bottom, left:right]  # n2ta3 el face l wahdo, region of interest 
            label = "without mask"  # default lw mafesh detection
            if face_roi.size != 0: # 34an at2aked bas en cropped box msh fady
                results = model(face_roi, stream=True) # detect mask ,, stream=True is used to iterate over the results.
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])  # class index (0 or 1)
                        label = model.names[cls]  # esm el class: with mask / without mask
                        break
            mask_status.append((left, top, right, bottom, name, label))  # save kol 7aga

    process_this_frame += 1  # zwd el counter

    # Draw results
    for (left, top, right, bottom, name, label) in mask_status:
        if name == "Unknown":
            text = f"A stranger is {label}"  # lw msh ma3roof
            color = (0, 255, 0) if label == "with mask" else (0, 0, 255)  # green lw 3leh mask, red lw la2
        else:
            text = f"{name} is {label}"  # el person bel esm da 3leh mask aw la2
            color = (0, 255, 0) if label == "with mask" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)  # rsm box 3la el face
        cv2.putText(frame, text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # ktaba el text fo2 el box

    cv2.imshow("Face Mask Recognition", frame)  # 3rd el frame 3la el screen
    if cv2.waitKey(1) & 0xFF == ord('q'):  # lw da5alt 'q' y2fel, w bget exactly last 8 bit.
        break

video.release()  # y2fel el camera
cv2.destroyAllWindows()  # y2fel kol el windows el mftoo7a
