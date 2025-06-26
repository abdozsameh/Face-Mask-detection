import cv2
import face_recognition
import pickle

with open('face_encodings.pkl', 'rb') as f:
    data = pickle.load(f)

known_encodings = data['encodings']
known_names = data['names']

video = cv2.VideoCapture(0)
video.set(3, 1280)
video.set(4, 720)

process_this_frame = 0
face_locations = []
face_names = []

print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    #processing
    if process_this_frame % 30 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            if True in matches:
                matched_index = matches.index(True)
                name = known_names[matched_index]

            face_names.append(name)

    process_this_frame += 1

    #The drawing of the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (150, 90, 3), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows() 