import cv2
import face_recognition
import os
import pickle
from halo import Halo


IMG_DIR = "D:\PROJECTS\Attendance_fr\Faces"
ENCODINGS_DIR = "D:\PROJECTS\Attendance_fr\FaceEncodings"

if not os.path.isdir(ENCODINGS_DIR):
    os.mkdir(ENCODINGS_DIR)

# use images to create encodings for faces

print('Processing images...')
names, face_encodings = [], []
num_people, image_processed = 0, 0

for folder in os.listdir(IMG_DIR):
    face_encoding = []

    for filename in os.listdir(f'{IMG_DIR}/{folder}'):
        if filename.rsplit('.')[-1] in ['jpg', 'jpeg', 'png']:
            image = face_recognition.load_image_file(f'{IMG_DIR}/{folder}/{filename}')
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) > 0:
                face_encoding.append(face_recognition.face_encodings(image, known_face_locations=face_locations)[0])

            image_processed += 1

    names.append(folder)
    face_encodings.append(face_encoding)

    num_people += 1

print(f'{num_people} people, {image_processed} images processed!')

# save the encodings for all faces within their respective directories

for name, face_encoding in zip(names, face_encodings):
    if not os.path.isdir(f'{ENCODINGS_DIR}/{name}'):
        os.mkdir(f'{ENCODINGS_DIR}/{name}')

    for index, encoding in enumerate(face_encoding):
        with open(f'{ENCODINGS_DIR}/{name}/{name}_{index}.pkl', 'wb') as fptr:
            pickle.dump(encoding, fptr)
