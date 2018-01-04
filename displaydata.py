import numpy as np
import cv2
import time
import pandas as pd
from collections import Counter
from random import shuffle

trainingData = np.load("training_data.npy")

def displayData(dataToDisplay):
    for data in dataToDisplay:
        image = data[0]
        choice = data[1]
        time.sleep(1)
        print(choice)
        cv2.imshow('test', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def balanceData(dataToBalance):
    df = pd.DataFrame(dataToBalance)
    print(df.head())
    print(Counter(df[1].apply(str)))
    print("Length Training Data before balacing: {}".format(len(dataToBalance)))

    lefts = []
    rights = []
    forwards = []

    for data in dataToBalance:
        img = data[0]
        choice = data[1]

        if choice == [0,1,0]:
            forwards.append([img,choice])

        elif choice == [0,0,1]:
            rights.append([img,choice])

        elif choice == [1,0,0]:
            lefts.append([img,choice])


    forwards = forwards[:len(lefts)][:len(rights)]
    lefts = lefts[:len(forwards)]
    rights = rights[:len(forwards)]

    finalData = forwards + lefts + rights + noops
    shuffle(finalData)
    print("Length Training Data after balancing: {}".format(len(finalData)))
    np.save("training_data_balanced.npy",finalData)

balanceData(trainingData)
#displayData(trainingData)
