import cv2
import os
import subprocess
def capture_photos(save_directory, desired_name, num_photos=5):
    # Create a folder with the desired name
    save_directory = os.path.join(save_directory, desired_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Open a connection to the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Counter for naming the captured photos
    count = 1

    while count <= num_photos:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Capture', frame)

        # Wait for a key press and check if it's the space key (key code 32)
        key = cv2.waitKey(1)
        if key == 32:  # space key pressed
            # Save the captured frame with the desired name
            photo_path = os.path.join(save_directory, f"{desired_name}_{count}.jpg")
            cv2.imwrite(photo_path, frame)

            print(f"Photo {count} captured and saved as {photo_path}")
            count += 1

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    # Run the face_encodings.py script
    subprocess.run(["/home/ubuntu/Desktop/esiot/myenv/bin/python3", "/home/ubuntu/Desktop/esiot/Attendance-Using-Face-Recognition/utils/face_encodings.py"])

if __name__ == "__main__":
    save_directory = "/home/ubuntu/Desktop/esiot/Attendance-Using-Face-Recognition/Faces"
    
    # Ask the user for the desired name
    desired_name = input("Enter the desired name for the photos: ")

    # Corrected variable name to num_photos
    num_photos = 5

    capture_photos(save_directory, desired_name, num_photos)
