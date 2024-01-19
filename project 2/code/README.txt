READ ME

RUNNING PREDICTIONS ON PREMADE MODELS

To run my code, have one file with all training data titled ‘ECE5242Proj2-train’, and one folder with all the test data titled 'ECE5242Proj2-test'

Run predict.py to predict gestures for all files within 'ECE5242Proj2-test'. This will load my pre-trained HMM models as well as a pre-trained KMeans from the folder 'model' for each gesture and print out the top 3 predictions with a confidence level.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STARTING FROM SCRATCH

Place all training data within a folder labeled ‘ECE5242Proj2-train’ and all test data within a folder labeled 'ECE5242Proj2-test'. 

Create an empty folder labeled 'model'.

1.
Running cluster.py will load all the training data, and perform KMeans clustering. It will save the trained KMeans model, and graph an example of raw data clustering on WaveData

2.
Running HMM.py will take the training data, and train HMM models for each gesture. After the training is complete it will save a transmission and emission matrix to the 'model' folder.

3.
Running predict.py will make predictions on the test data. The output will print out the top 3 predictions with a confidence level.