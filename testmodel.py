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


while(True):
    # Capture frame-by-frame
    ret, webcamImage = cap.read()

    imageToPredict = cv2.resize(webcamImage, (80, 60))
    imageToPredict = cv2.cvtColor(imageToPredict, cv2.COLOR_BGR2GRAY)

    prediction = model.predict([imageToPredict.reshape(WIDTH,HEIGHT,1)])[0]
    print(prediction)

    cv2.imshow('frame', imageToPredict)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()