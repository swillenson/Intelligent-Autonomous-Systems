import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import scipy
import pickle

#define all my personal folder paths, these can be changed if necessary
trainPath = "ECE5242Proj2-train/"
testPath = "ECE5242Proj2_train_additional/"
dir_list = os.listdir(trainPath)
modelFolder = "model/"

beat3Data = []
beat4Data = []
circleData =[]
eightData = []
infData = []
waveData = []

#load all of my data and sort into lists for each gesture
for file in dir_list[:]:
    filePath = os.path.join(trainPath, file)
    data = np.loadtxt(filePath)
    data = np.asarray(data)
    if file[0] == 'b':
        if file[4] == '3':
            beat3Data.append(data)

        if file[4] == '4':
            beat4Data.append(data)
    
    elif file[0] == 'c':
        circleData.append(data)

    elif file[0] == 'e':
        eightData.append(data)

    elif file[0] == 'i':
        infData.append(data)

    elif file[0] == 'w':
        waveData.append(data)

fullData = [beat3Data, beat4Data, circleData, eightData, infData, waveData]
fullDataForCluster = []
dummy = [0,0,0,0,1,1,1,1]
#visualize data
for data in fullData:
    for datum in data:
        fullDataForCluster.append(datum)
dataConcat = np.concatenate(fullDataForCluster)

with open('model/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

singleDataConcat = np.concatenate(beat3Data, axis = 0)
# kmeans.fit(singleDataConcat[:, 1:])
observations = kmeans.predict(singleDataConcat[:, 1:])




#data types:beat3Data, beat4Data, circleData, eightData, infData, waveData 

dataFull = [beat3Data, beat4Data, circleData, eightData, infData, waveData]

HMMmodel = []


numHiddenStates = 10
numObservationClusters = 60
numGestures = 6
HMMmodels = []

#data types:beat3Data, beat4Data, circleData, eightData, infData, waveData 


#initialize random A values
#use the left right matrix to initialize A, use .5 for each values down/right
#column should add to 1
A = np.random.rand(numHiddenStates, numHiddenStates)
A /= np.sum(A, axis = 0)

#trying to initialize A better

# A = np.zeros((numHiddenStates, numHiddenStates))
# for i in range(numHiddenStates):
#     A[i,i] = 0.5
#     if i == numHiddenStates - 1:
#         break
#     A[i + 1, i] = 0.5

# print(A)

#initialize random B values
#want to normalize by looping through each value and diving by the sum of row
B = np.random.rand(numHiddenStates, numObservationClusters) 
B /= np.sum(B, axis = 0)

#initialize pi values
#just [1, 0, 0 ....]
pi = np.zeros(numHiddenStates)
pi[0] = 1

def fwdAlgo(data, A, B, pi):
    numObservations = len(data)
    numHiddenStates, numObservationClusters= B.shape
    alpha = np.zeros((numObservations, numHiddenStates))
    scale = np.zeros(numObservations)
    alpha[0, :] = pi * B[:, data[0]]
    scale[0] = 1 / np.sum(alpha[0, :])
    alpha[0, :] = alpha[0, :] * scale[0]


    for t in range(1, len(data)):
        for i in range(numHiddenStates):
            alpha[t, i] =  np.dot(alpha[t - 1, :], A[i, :] * B[i, data[t]])
        scale[t] = 1 / np.sum(alpha[t, :])
        alpha[t, :] = scale[t] * alpha[t]

    likelihood =  -np.sum(np.log(scale))
    # print(scale)
    print('train', likelihood)
    # print('alpha', alpha)

    return alpha, scale, likelihood

def backAlgo(data, A, B, scale):
    numHiddenStates, numObservationClusters = B.shape
    numObservations = len(data)
    beta = np.zeros((numObservations, numHiddenStates))
    beta[numObservations - 1, :] = 1 * scale[-1]
    for t in range(numObservations - 2, -1, -1):
        for i in range(numHiddenStates):

            beta[t, i] = np.longdouble(np.dot( A[:, i], B[:, data[t + 1]] * beta[t + 1, :]))
        beta[t, :] = scale[t] * beta[t, :]
        
    # print('beta', beta)
    # print('scale', scale)
    return beta

def Estep(data, A, B, alpha, beta):
    numHiddenStates, numObservationClusters = B.shape
    numObservations = len(data)
    gamma = np.zeros((numObservations, numHiddenStates))

    for t in range(numObservations):
        for i in range(numHiddenStates):
            gamma[t, i] = (alpha[t, i] * beta[t, i]) / np.dot(alpha[t, :], beta[t, :])

    xi = np.zeros((numObservations, numHiddenStates, numHiddenStates))

    for t in range(numObservations - 1):
        denom = 0
        for i in range(numHiddenStates):
            for j in range(numHiddenStates):
                denom += alpha[t, i] * A[j, i] * B[j, int(data[t + 1])] * beta[t + 1, j]
        
        for i in range(numHiddenStates):
            for j in range(numHiddenStates):
                num = alpha[t, i] * A[j, i] * B[j, int(data[t + 1])] * beta[t + 1, j]
                xi[t, i, j] = num / denom
    # print('gamma', gamma)
    # print('xi', xi)
    return gamma, xi

def Mstep(data, A, B, gamma, xi):
    numHiddenStates, numObservationClusters = B.shape
    numObservations = len(data)
    for i in range(numHiddenStates):
        for j in range(numHiddenStates):
            num = 0
            denom = np.sum(xi[:-1, i, :])
            # denom = 0
            # for t in range(numObservations - 1):
            #     for j in range(numHiddenStates):
            #         denom += xi[t, i, j]
            for t in range(numObservations - 1):
                num += xi[t, i, j]

            A[j, i] = num / denom

    for i in range(numHiddenStates):
        for o in range(numObservationClusters):
            num = 0
            for t in range(numObservations):
                if data[t] == o:
                    num += gamma[t, i]
                # if num == 0:
                #     B[i, o] = 1e-8
            B[i, o] = num / np.sum(gamma[:, i])
    # B += 1e-10
    return A, B
                

def baumWelch(data, A, B, pi, numIterations):
    for i in range(numIterations):
        alpha, scale, likelihood = fwdAlgo(data, A, B, pi)
        beta = backAlgo(data, A, B, scale)
        gamma, xi = Estep(data, A, B, alpha, beta)
        A, B = Mstep(data, A, B, gamma, xi)
        # print('A', A)
        # print('B', B)
        # print('gamma', gamma)
        # print('alpha', alpha) 
        # print('beta', beta)

    return A, B, likelihood, scale






A, B, likelihood, scale = baumWelch(observations, A, B, pi, 10)
# print('A', A)
# print('B', B)
# print('P', likelihood)
# print(labels.shape)
# np.save(modelFolder + 'waveA', A)
# np.save(modelFolder + 'waveB', B)
# np.save(modelFolder + 'wavePi', pi)
# np.save(modelFolder + 'waveProb', likelihood)
# np.save(modelFolder + 'waveScale', scale)

# Save A matrix
with open("model/beat3A_final.pkl", "wb") as f:
    pickle.dump(A, f)

# save B matrix
with open("model/beat3B_final.pkl", "wb") as f:
    pickle.dump(B, f)



#Testing

beat3DataTest = []
beat4DataTest = []
circleDataTest =[]
eightDataTest = []
infDataTest = []
waveDataTest = []

dir_list = os.listdir(testPath)

for file in dir_list[:]:
    filePath = os.path.join(testPath, file)
    data = np.loadtxt(filePath)
    if file[0] == 'b':
        if file[4] == '3':
            beat3DataTest.append(data)

        if file[4] == '4':
            beat4DataTest.append(data)
    
    elif file[0] == 'c':
        circleDataTest.append(data)

    elif file[0] == 'e':
        eightDataTest.append(data)

    elif file[0] == 'i':
        infDataTest.append(data)

    elif file[0] == 'w':
        waveDataTest.append(data)



dataTestConcat = np.concatenate(waveDataTest, axis = 0)

observationsTest = kmeans.predict(dataTestConcat[:, 1:])
numObservations = len(observationsTest)



def fwdAlgoTest(data, A, B, pi, scale):
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
    print('test', likelihood)
    # print('alpha', alpha)
    # print('test', A)
    # print('test', B)


    return likelihood

p = fwdAlgoTest(observationsTest, A, B, pi, scale)
print(p)


# print('A', A)
# print('B', B)