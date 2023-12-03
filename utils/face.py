import cv2
import face_recognition
import os
import time

IMG_DIR = "D:\PROJECTS\Attendance_fr\Images"
SAVE_DIR = 'D:\PROJECTS\Attendance_fr\Faces'

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

print('Extracting faces...')
faces_extracted = 0

for filename in os.listdir(IMG_DIR):
    extension = filename.rsplit('.')[-1] if '.' in filename else None

    if extension in ['jpg', 'png']:
        image = face_recognition.load_image_file(f'{IMG_DIR}/{filename}')
        face_locations = face_recognition.face_locations(image)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = cv2.cvtColor(image[top:bottom+1, left:right+1], cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{SAVE_DIR}/IMG_{time.time_ns()}.{extension}', face_image)
            faces_extracted += 1

print(f'{faces_extracted} faces extracted!')
