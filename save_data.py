import numpy as np
from random import shuffle

trainingData = []
file_name = 'training_data.npy'

def captureData(image,choice):

    global trainingData
    trainingData.append([image, choice])
    print(len(trainingData))

    if len(trainingData) == 1000:
        shuffle(trainingData)
        np.save(file_name,trainingData)
        print('===============================SAVED===========================')
