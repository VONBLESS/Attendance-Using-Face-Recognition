import cv2
import face_recognition
import keyboard
import os
import pickle
from halo import Halo
from datetime import datetime


ENCODINGS_DIR = "./FaceEncodings"
TOLERANCE = 0.5

# check if web-cam is available or not? how to do so?, then process images.

spinner = Halo(spinner="dots", placement="right")

# launch the webcam first
spinner.text = "Launching web cam"
spinner.start()
videoCapture = cv2.VideoCapture(0)
spinner.stop()

if videoCapture.isOpened():
    print("Web-cam launched!")
else:
    print("Error while launching web-cam!")
    exit(1)

# read pre-computed face encodings

names, face_encodings = [], []
num_people, num_encodings = 0, 0

spinner.text = "Loading saved (labelled) face encodings"
spinner.start()

for folder_name in os.listdir(ENCODINGS_DIR):
    face_encoding = []
    for filename in os.listdir(f"{ENCODINGS_DIR}/{folder_name}"):
        with open(f"{ENCODINGS_DIR}/{folder_name}/{filename}", "rb") as fptr:
            face_encoding.append(pickle.load(fptr))

        num_encodings += 1

    names.append(folder_name)
    face_encodings.append(face_encoding)

    num_people += 1

spinner.stop()
print(f"{num_people} people are found, {num_encodings} face encodings are loaded!")


def stop_running():
    global running
    running = False
    keyboard.send("\b")

def markAtt(name):
    with open('Attendance.csv','r+') as f:
        Data = f.readlines()
        nameList = []
        for line in Data:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            date_time_string = now.strftime('%H : %M : %S')
            f.writelines(f'\n{name},{date_time_string}')

running = True

keyboard.add_hotkey("esc", stop_running, suppress=True)

while running:
    _, frame = videoCapture.read()
    # flip the image horizontally
    # cv2.flip(frame, 1, frame)

    text_labels = []
    boxes = []
    identified_names = []

    face_locations = face_recognition.face_locations(frame)

    # use K-D Tree? 128-dimensional output from the face_recognition, similar to latent space?
    for face_location in face_locations:
        top, right, bottom, left = face_location
        boxes.append([(left, top), (right, bottom)])

        scale = abs((right - left) * (bottom - top)) ** 0.5

        unknown_face_encoding = face_recognition.face_encodings(
            frame, known_face_locations=[face_location]
        )[0]
        face_distances = map(
            lambda fe: min(face_recognition.face_distance(fe, unknown_face_encoding)),
            face_encodings,
        )
        guessed_names = sorted(zip(face_distances, names))

        text_labels.append(
            [
                guessed_names[0][1]
                if guessed_names[0][0] < TOLERANCE
                else "unknown_person",
                (left, bottom),
                scale,
            ]
        )

        if guessed_names[0][0] < TOLERANCE:
            identified_names.append(guessed_names[0][1])
            markAtt(guessed_names[0][1])


    for label in text_labels:
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        fontScale = label[2] / cv2.getTextSize(label[0], font, 2, 1)[0][0] * 2
        w, h = cv2.getTextSize(label[0], font, fontScale, 2)[0]
        cv2.rectangle(
            frame,
            (label[1][0], label[1][1] + 2),
            (label[1][0] + w, label[1][1] + int(h * 1.5) + 2),
            [0] * 3,
            -1,
        )
        cv2.putText(
            frame,
            label[0],
            (label[1][0], label[1][1] + h),
            font,
            fontScale,
            [255] * 3,
            2,
        )

    for box in boxes:
        cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 2)

    cv2.imshow("WebCam", frame)

    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("WebCam", 1) <= 0:
        break

    identified_names = list(set(identified_names))
    identified_names.sort()

    print(f'\x1b[2K\r{identified_names if len(identified_names) > 0 else ""}', end="")


videoCapture.release()
cv2.destroyAllWindows()

