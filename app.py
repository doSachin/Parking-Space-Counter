import cv2
import pickle
import cvzone
import os
import numpy as np
import pdb
from flask import Flask, render_template, request, Response

app = Flask(__name__)

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

# to display parking area
IMG_FOLDER = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48


def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        # cv2.imshow(str(x * y), imgCrop)
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(imgPro, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(imgPro, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(imgPro, f"Total: {len(posList)}              Available: {spaceCounter}", (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))


def gen_frames():  # generate frame by frame from camera
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame

        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        checkParkingSpace(imgDilate)

        cv2.imshow("Image", frame)
        cv2.waitKey(10)

        ret, jpeg = cv2.imencode('.jpg', checkParkingSpace(imgDilate))

        return jpeg.tobytes()


@app.route('/')
def index():
    parkingSpace = os.path.join(app.config['UPLOAD_FOLDER'], 'carParkImg.png')
    return render_template("index.html", user_image=parkingSpace)


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
