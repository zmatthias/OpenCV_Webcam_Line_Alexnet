import numpy as np
import cv2
import save_data

cap = cv2.VideoCapture(0)
xDifference = 0.0

def ProcessImage(passedImage):

    processedImage = cv2.cvtColor(passedImage, cv2.COLOR_BGR2HSV)
    lowerBlue = np.array([108, 20, 15])
    upperBlue = np.array([130, 255, 255])
    mask = cv2.inRange(processedImage, lowerBlue, upperBlue)
    processedImage = cv2.bitwise_and(passedImage, passedImage, mask=mask)
    processedImage = cv2.medianBlur(processedImage,13)

    processedImage = cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY)

    processedImage = cv2.Canny(processedImage, threshold1=20, threshold2=100)
    processedImage = cv2.GaussianBlur(processedImage,(7,7),0)
    lines = cv2.HoughLinesP(processedImage, 1, np.pi/180, 10 ,np.array([]), 200, 20)
    return lines
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
    if (xDifference > 4):
        return [0,1,0]             #straight,left,right
    if (xDifference < -4):
        return [0,0,1]
    else:
        return [1,0,0]

while(True):
    # Capture frame-by-frame
    ret, webcamImage = cap.read()
    lines = ProcessImage(webcamImage)
    imageToShow = DrawLines(lines,webcamImage)
    print(xDifference)
    choice = Decide(xDifference)
    print(choice)

    imageToSave = cv2.resize(webcamImage, (80, 60))
    imageToSave = cv2.cvtColor(imageToSave, cv2.COLOR_BGR2GRAY)
    save_data.captureData(imageToSave,choice)

    cv2.imshow('frame', imageToShow)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()