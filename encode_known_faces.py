import os
import face_recognition
import pickle


KNOWN_FACES_DIR = 'known faces'
ENCODINGS_FILE = 'face_encodings.pkl'

known_encodings = [] # empty list to store face encodings
known_names = [] # empty list to store names of known faces

for name in os.listdir(KNOWN_FACES_DIR): # ba3mel looping through each person in the directory elly hya known faces
    person_folder_path = os.path.join(KNOWN_FACES_DIR, name) # ba3mel path lel person
    if not os.path.isdir(person_folder_path): # 34an lw file msh folder aw msh directory ye3ml skip 
        continue

    for filename in os.listdir(person_folder_path): # ba3mel looping through each image in the person directory, filename ely howa jpg
        imagepath = os.path.join(person_folder_path, filename) 
        image = face_recognition.load_image_file(imagepath) 
        encodings = face_recognition.face_encodings(image) # ba3mel load image and get face encodings

        if encodings:
            known_encodings.append(encodings[0]) # ba3mel add llencoding we {0} 34an howa kaza encoding lel sha5s bas howa encoding wa7ed lel image
            known_names.append(name) # ba3mel add name to list of known names
            print(f"Encoded {filename} for {name}")
        else:
            print(f"No face found in {filename}")

data = {'encodings': known_encodings, 'names': known_names} 

with open(ENCODINGS_FILE, 'wb') as file:
    pickle.dump(data, file)

print(f"\n✅ Encoding complete. Saved to '{ENCODINGS_FILE}'")