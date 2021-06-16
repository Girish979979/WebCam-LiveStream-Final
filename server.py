# Author : Girish Kumar Reddy Veerepalli
# File Name : server.py

"""
This file contains the logic required to process the webcam feed and display it on the target browser or window.
The Image sent is received and processed.
Facial recognition logic has been added.
Flask has been used to handle the browser part of the program
"""

# import the necessary packages
import imagezmq
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

# Initialize the cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def gen_frames():
    """
    This function receives the frames from the client and processes them and displays them.
    It also handles the functionality of detecting faces and drawing bounding boxes around them.
    :return:
    """
    while True:

        (rpiName, frame) = imageHub.recv_image()

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use the cascade classifier to detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the video frames on the window
        cv2.imshow('Output',frame)

        # Encoding
        ret, buffer = cv2.imencode('.jpg', frame)

        # Break is no frame is found
        if not ret:
            break

        # Convert to bytes and display the image.
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Sample reply back to the Image Hub
        imageHub.send_reply(b'OK')

        # detect any key presses
        key = cv2.waitKey(1) & 0xFF

        # Break if the key pressed = q
        if key == ord("q"):
            break

# Flask part of the application
@app.route('/video_feed')
def video_feed():
    """
    Video Streaming Routing.
    This would be in the src attribute of the img tag.
    :return: Response()
    """

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """
    Video streaming home page.
    The template is taken from the index.html file
    """
    return render_template('index.html')


if __name__ == '__main__':
    """
    The main program that runs the application
    """
    app.run(debug=True,threaded=True, use_reloader = False)

# Clean up
cv2.destroyAllWindows()