from flask import Flask, render_template, Response, request, redirect, url_for
import cv2 as cv
import numpy as np
import os
import hand_tracking_module as htm

app = Flask(__name__)

folderPath = "Painting/Header"
myList = os.listdir(folderPath)
myList.sort()
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    if image is None:
        print(f"Failed to load image: {imPath}")
    overlayList.append(image)

header = overlayList[0]
drawColor = (0, 0, 255)  # Default to red
brushThickness = 15
eraserThickness = 100

cap = None
detector = htm.handDetector(detectionCon=0.85, maxHands=1)
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
process = True

def initialize_capture():
    global cap
    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

initialize_capture()

def generate_frames():
    global header, drawColor, xp, yp, imgCanvas, process, cap

    while process:
        success, img = cap.read()
        if not success:
            break
        else:
            img = cv.flip(img, 1)
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                fingers = detector.fingerUp()

                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (255, 0, 255), cv.FILLED)
                    if y1 < 125:
                        if 250 < x1 < 450:
                            header = overlayList[0]
                            drawColor = (0, 0, 255)  # Red
                        elif 550 < x1 < 750:
                            header = overlayList[1]
                            drawColor = (0, 255, 255)  # Blue
                        elif 800 < x1 < 950:
                            header = overlayList[2]
                            drawColor = (0, 255, 0)  # Green
                        elif 1050 < x1 < 1200:
                            header = overlayList[3]
                            drawColor = (0, 0, 0)  # Eraser
                    cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)

                if fingers[1] and fingers[2] == False:
                    if drawColor == (0, 0, 0):
                        cv.circle(img, (x1, y1), 15, (169, 169, 169), cv.FILLED)
                    cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if drawColor == (0, 0, 0):
                        cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    else:
                        cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    xp, yp = x1, y1

            imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
            _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
            imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
            img = cv.bitwise_and(img, imgInv)
            img = cv.bitwise_or(img, imgCanvas)
            img[0:125, 0:1280] = header

            ret, buffer = cv.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/restart', methods=['POST'])
def restart():
    global process, imgCanvas, xp, yp
    process = False
    cap.release()
    cv.destroyAllWindows()
    
    # Reset variables
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    xp, yp = 0, 0
    process = True
    initialize_capture()

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
