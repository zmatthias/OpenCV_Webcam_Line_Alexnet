from __future__ import print_function
import numpy as np
import cv2
from alexnet import alexnet

cap = cv2.VideoCapture(0)

WIDTH  = 80
HEIGHT = 60
LR = 1e-3

MODEL_NAME = 'alexnet.model'
model = alexnet(WIDTH,HEIGHT,LR)
model.load(MODEL_NAME)


def PreProcessImage(passedImage):
    processedImage = cv2.cvtColor(passedImage, cv2.COLOR_BGR2HSV)
    lowerBlue = np.array([90, 70, 50])
    upperBlue = np.array([100, 255, 210])
    mask = cv2.inRange(processedImage, lowerBlue, upperBlue)
    mask = cv2.erode(mask, None, iterations=4)
    processedImage = cv2.dilate(mask, None, iterations=7)
    return processedImage

def printPrediction(prediction):

    forwardPrediction = prediction[1]
    turningPrediction = prediction[0] - prediction[2]

    print("Forward: {} \t \t \t \t \t Turning: {} ".format(format(forwardPrediction, '.2f'), format(turningPrediction, '.2f')))

while(True):

    # Capture frame-by-frame
    ret, webcamImage = cap.read()

    preProcessedImage = PreProcessImage(webcamImage)
    imageToPredict = cv2.resize(preProcessedImage, (80, 60))
    prediction = model.predict([imageToPredict.reshape(WIDTH,HEIGHT,1)])[0]
    print(prediction)
    printPrediction(prediction)

    cv2.imshow('frame', imageToPredict)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

