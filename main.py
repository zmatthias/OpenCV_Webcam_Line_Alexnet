import numpy as np
import cv2
import save_data

cap = cv2.VideoCapture(0)
xDifference = 0.0

def FindLines(passedImage):
    processedImage = cv2.Canny(passedImage,100,200)
    processedImage = cv2.GaussianBlur(processedImage,(3,3),0)
    lines = cv2.HoughLinesP(passedImage, 20, np.pi/180, 10 ,np.array([]), 200, 20)
    return lines

def PreProcessImage(passedImage):
    processedImage = cv2.cvtColor(passedImage, cv2.COLOR_BGR2HSV)
    lowerBlue = np.array([90, 70, 50])
    upperBlue = np.array([100, 255, 210])
    mask = cv2.inRange(processedImage, lowerBlue, upperBlue)
    mask = cv2.erode(mask, None, iterations=4)
    processedImage = cv2.dilate(mask, None, iterations=7)
    return processedImage


def DrawLines(lines,passedImage):
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:

                #Problem: Anfangs und Endpunkte der Linien werden geflippt, sodass der Endpunkt immer rechts ist
                #dadurch x1-x2 als Richungsvektor nicht moeglich
                if y2 >= y1:        #deflipping
                    x1_old = x1
                    x1 = x2
                    x2 = x1_old
                    y1_old = y1
                    y1 = y2
                    y2 = y1_old

                cv2.line(passedImage, (x1, y1), (x2, y2), (0, 255, 0), 10)
                cv2.line(passedImage, (x1, y1), (x1, y1), (255, 0, 0), 10)
                cv2.line(passedImage, (x2, y2), (x2, y2), (0, 0, 255), 10)

                #print(len(lines[0]))
                global xDifference
                xDifference += x1-x2
                xDifference = xDifference/len(lines[0])
    except:
        pass
    return passedImage

def Decide(xDifference):

    if (xDifference > 3):
        return [1,0,0]             #straight,left,right
    if (xDifference < -3):
        return [0,0,1]
    else:
        return [0,1,0]

while(True):
    # Capture frame-by-frame
    ret, webcamImage = cap.read()
    imageToSave = webcamImage
    webcamImage = cv2.resize(webcamImage,(600,480))

    preProcessedImage = PreProcessImage(webcamImage)
    lines = FindLines(preProcessedImage)
    imageToShow = DrawLines(lines,webcamImage)

    if (lines is None):
        choice = [0,0,0]
    else:
        choice = Decide(xDifference)

    print(choice)
    imageToSave = cv2.resize(preProcessedImage, (80, 60))
   # imageToSave = cv2.cvtColor(imageToSave, cv)
    save_data.captureData(imageToSave,choice)

    cv2.imshow('frame', imageToShow)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()