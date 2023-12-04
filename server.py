from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# OpenCV VideoCapture object for the camera (0 represents the default camera)
cap = cv2.VideoCapture(1)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            # Convert the image to bytes and yield it
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
    # return "hell"
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000, debug=True)
