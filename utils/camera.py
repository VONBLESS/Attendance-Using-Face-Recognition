import cv2
import time

vc = cv2.VideoCapture(0)

if vc.isOpened():
    while True:
        start_time = time.time()
        _, frame = vc.read()

        # flip selfie?
        # cv2.flip(frame, 1, frame)

        # resize image?
        # frame = cv2.resize(frame, (320, 240))

        cv2.imshow('WebCam', frame)

        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('WebCam', 1) <= 0:
            break

        end_time = time.time()

        if end_time > start_time:
            print(f'\rfps : {1 / (end_time-start_time)}', end='')
else:
    print('Error while launching web-cam!')

vc.release()
cv2.destroyAllWindows()
