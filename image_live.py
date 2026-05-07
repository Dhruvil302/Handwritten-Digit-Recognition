import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
from scipy import ndimage
import math

model = load_model('CNN_model.h5')

def predict(image):
    input = cv2.resize(image, (28, 28)).reshape((28, 28, 1)).astype('float32') / 255.0
    return model.predict_classes(np.array([input]))

def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty

def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(img, M, (cols, rows))

cap = cv2.VideoCapture(0)
canvas = np.ones((200, 200), dtype="uint8") * 255
canvas[0:200, 0:200] = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)
    cv2.imshow("Test Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(255 - img, (28, 28))
        _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        while np.sum(gray[0]) == 0:
            gray = gray[1:]
        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)
        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]
        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)

        rows, cols = gray.shape
        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            gray = cv2.resize(gray, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            gray = cv2.resize(gray, (cols, rows))

        colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
        rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

        shiftx, shifty = getBestShift(gray)
        gray = shift(gray, shiftx, shifty)

        result = predict(gray)
        canvas[0:200, 0:200] = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, str(result), (55, 110), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
