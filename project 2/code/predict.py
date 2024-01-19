import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import pickle



trainPath = "ECE5242Proj2-train/"
testPath = "ECE5242Proj2-test/"
dir_list = os.listdir(trainPath)
modelFolder = "model/"



#Testing 

beat3DataTest = []
beat4DataTest = []
circleDataTest =[]
eightDataTest = []
infDataTest = []
waveDataTest = []

dir_list = os.listdir(testPath)




numHiddenStates = 10
numObservationClusters = 60
numGestures = 6

def fwdAlgoTest(data, A, B, pi):
    numObservations = len(data)
    numHiddenStates, numObservationClusters= B.shape
    alpha = np.zeros((numObservations, numHiddenStates))
    scale = np.zeros(numObservations)
    alpha[0, :] = pi * B[:, data[0]]
    if np.sum(alpha[0, :]) < 1e-10:
        scale[0] = 1 / 1e-10
    else:
        scale[0] = 1 / np.sum(alpha[0, :])
    alpha[0, :] = alpha[0, :] * scale[0]

    for t in range(1, len(data)):
        for i in range(numHiddenStates):
            alpha[t, i] =  np.dot(alpha[t - 1, :], A[:, i]) * B[i, data[t]]
        if np.sum(alpha[t, :]) < 1e-10:
            scale[t] = 1 / 1e-10
        else:
            scale[t] = 1 / np.sum(alpha[t, :])
        alpha[t, :] = scale[t] * alpha[t]

    likelihood =  -np.sum(np.log(scale))
    # print('test', likelihood)
    # print('alpha', alpha)
    # print('test', A)
    # print('test', B)


    return likelihood




pi = np.zeros(numHiddenStates)
pi[0] = 1
with open('model/waveA_final.pkl', 'rb') as f:
    Awave = pickle.load(f)
with open('model/waveB_final.pkl', 'rb') as f:
    Bwave = pickle.load(f)
with open('model/infA_final.pkl', 'rb') as f:
    Ainf = pickle.load(f)
with open('model/infB_final.pkl', 'rb') as f:
    Binf = pickle.load(f)
with open('model/eightA_final.pkl', 'rb') as f:
    Aeight = pickle.load(f)
with open('model/eightB_final.pkl', 'rb') as f:
    Beight = pickle.load(f)
with open('model/circleA_final.pkl', 'rb') as f:
    Acircle = pickle.load(f)
with open('model/circleB_final.pkl', 'rb') as f:
    Bcircle = pickle.load(f)
with open('model/beat4A_final.pkl', 'rb') as f:
    Abeat4 = pickle.load(f)
with open('model/beat4B_final.pkl', 'rb') as f:
    Bbeat4 = pickle.load(f)
with open('model/beat3A_2.pkl', 'rb') as f:
    Abeat3 = pickle.load(f)
with open('model/beat3B_2.pkl', 'rb') as f:
    Bbeat3 = pickle.load(f)

with open('model/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

As = [Awave, Ainf, Aeight, Acircle, Abeat4, Abeat3]
Bs = [Bwave, Binf, Beight, Bcircle, Bbeat4, Bbeat3]
fileList = []
gestureIndexList = ['Wave', 'Infinity', 'Eight', 'Circle', 'Beat4', 'Beat3']

for file in dir_list[:]:
    p = []
    fileList.append(str(file))
    test = []
    filePath = os.path.join(testPath, file)
    data = np.loadtxt(filePath)
    test.append(data)
    dataTestConcat = np.concatenate(test, axis = 0)
    observationsTest = kmeans.predict(dataTestConcat[:, 1:])
    numObservations = len(observationsTest)

    for i in range(len(As)):
        pCurrent = fwdAlgoTest(observationsTest, As[i], Bs[i], pi)
        p.append(pCurrent)
    enumeratedP = list(enumerate(p))
    sortedP = sorted(enumeratedP, key = lambda x:x[1], reverse = True)
    confidencePercentage = 1 - (sortedP[0][1] / (sortedP[0][1] + sortedP[1][1]))
    top3 = sortedP[:3]
    topFeatures = []
    for idx, val in top3:
        topFeatures.append(gestureIndexList[idx])
        # print(val)

        
    print('Predictions for', str(file), '1st:', topFeatures[0], ', 2nd:', topFeatures[1], ', 3rd:', topFeatures[2], ', with Confidence: ', confidencePercentage)




    # likelihoods = np.exp(p- np.max(p))
    # likelihoods /= np.sum(likelihoods)
    # maxIndex = np.argmax(p)
    # confidencePercentage = likelihoods[maxIndex] * 100




# p = fwdAlgoTest(observationsTest, A, B, pi, scale)
# print(p)



