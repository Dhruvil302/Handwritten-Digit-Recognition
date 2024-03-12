import cv2
from PIL import Image
import numpy as np
from scipy import ndimage
import math
from keras.models import load_model
img=cv2.imread("files3.jpg",0)
gray=cv2.resize(255-img,(28,28))
(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
while np.sum(gray[0]) == 0:
    gray = gray[1:]

while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)

rows,cols = gray.shape

if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    gray = cv2.resize(gray, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))

colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

shiftx,shifty = getBestShift(gray)
shifted = shift(gray,shiftx,shifty)
gray = shifted
data=Image.fromarray(gray)
data.save('test.png')

model=load_model('CNN_nodel.h5')
def predict(image):
    input = cv2.resize(image,(28,28)).reshape((28 , 28,1)).astype('float32') / 255.0
    return model.predict_classes(np.array([input]))
result=predict(gray)
str1=str(result)
str2="Prediction is :"
str_f=str2+str1
canvas = np.ones((500,500), dtype="uint8") * 255
canvas[0:500,0:500] = 0
img1 = canvas[0:500,0:500]
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1,str(str_f),(130,270),font,1,(255,0,0),2,cv2.LINE_AA)
while(True):
	cv2.imshow("Test Canvas", canvas)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break