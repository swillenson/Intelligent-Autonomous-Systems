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

# print(dataConcat.shape)
# plt.scatter(waveDataConcat[:, 0], waveDataConcat[:, 1])
# plt.show()
# dataConcat.flatten()

# Start KMeans clustering
# kmeans = KMeans(n_clusters =60)
# kmeans.fit(dataConcat[:, 1:])

# labels = kmeans.labels_
# with open('model/kmeans_model_v2.pkl', 'wb') as f:
#     pickle.dump(kmeans, f)
#     f.close()
# labels = dummy

with open('model/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)


waveDataConcat = np.concatenate(waveData, axis = 0)

waveConcat = np.concatenate(waveData, axis = 0)
waveObservations = kmeans.predict(waveConcat[:, 1:])

infConcat = np.concatenate(infData, axis = 0)
infObservations = kmeans.predict(infConcat[:, 1:])

eightConcat = np.concatenate(eightData, axis = 0)
eightObservations = kmeans.predict(eightConcat[:, 1:])

circleConcat = np.concatenate(circleData, axis = 0)
circleObservations = kmeans.predict(circleConcat[:, 1:])

beat3Concat = np.concatenate(beat3Data, axis = 0)
beat3Observations = kmeans.predict(beat3Concat[:, 1:])

beat4Concat = np.concatenate(beat4Data, axis = 0)
beat4Observations = kmeans.predict(beat4Concat[:, 1:])




#visualize clustered raw wave data

# fig, axes = plt.subplots(2, 2, figsize = (10, 10))

# waveDataConcat = np.concatenate(waveData, axis = 0)
# waveDataObservations = kmeans.predict(waveDataConcat[:, 1:])

# axes[0, 0].scatter(waveDataConcat[:, 1], waveDataConcat[:, 4], c=waveDataObservations, cmap='rainbow')
# axes[0, 0].set_xlabel('Ax')
# axes[0, 0].set_ylabel('Wx')
# axes[0, 0].set_box_aspect(1) 
# axes[0, 0].set_title('KMeans Clustering on Wave Gesture Data')



# axes[0, 1].scatter(waveDataConcat[:, 2], waveDataConcat[:, 5], c=waveDataObservations, cmap='rainbow')
# axes[0, 1].set_xlabel('Ay')
# axes[0, 1].set_ylabel('Wy')
# axes[0, 1].set_box_aspect(1)


# axes[1, 0].scatter(waveDataConcat[:, 3], waveDataConcat[:, 6], c=waveDataObservations, cmap='rainbow')
# axes[1, 0].set_xlabel('Az')
# axes[1, 0].set_ylabel('Wz')
# axes[1, 0].set_box_aspect(1)

# axes[1, 1].scatter(waveDataConcat[:, 1], waveDataConcat[:, 2], c=waveDataObservations, cmap='rainbow')
# axes[1, 1].set_xlabel('Ax')
# axes[1, 1].set_ylabel('Ay')
# axes[1, 1].set_box_aspect(1)

# fig.subplots_adjust(hspace=0.4)


# plt.show()


