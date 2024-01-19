import cv2
import matplotlib.pyplot as plt
import numpy as np
import roipoly
import os
from PIL import Image
from skimage import data, util
from skimage.measure import label, regionprops
import math

#import my training data
trainPath = "train/"
testPath = "test/"
dir_list = os.listdir(trainPath)

#change these paths if necessary
outputInsideFolder = "labeledInside/"
outputOutsideFolder = "labeledOutside/"

#load my .npys from my label.py
insidePixels = np.load(outputInsideFolder + 'insidePixels.npy')
outsidePixels = np.load(outputOutsideFolder + 'outsidePixels.npy')
imgTotalPixels = np.load(outputInsideFolder + 'imgTotalPixels.npy')

#recalculate mean and cov
meanInside = np.mean(insidePixels, axis = 0)
meanOutside = np.mean(outsidePixels, axis = 0)
covInside = np.cov(insidePixels.T)
covOutside = np.cov(outsidePixels.T)


#define my pdf function with Bayes Rule

def pdf(x, mean, cov):
    denom = 1/np.sqrt(2*np.pi * np.linalg.det(cov))
    num1 = np.matmul(-(x - mean).T, np.linalg.inv(cov))
    num2 = np.matmul(num1, (x - mean))
    return ((denom * np.exp(num2)))

def BayesRule(Pxy, Py, Px):
    return Pxy * Py / Px

#y1 = orange cone, y2 = not cone; these will be used for Bayes Rule
Py1 = len(insidePixels) / (len(insidePixels) + len(outsidePixels))
Py2 = len(outsidePixels) / (len(insidePixels) + len(outsidePixels))

#initialize my lookup table, RGBTable, with 0's in a 3D array
#I loop over each possible RGB value and calculate the pdf on it so I can save time in the train step
RGBTable = np.zeros((64, 64, 64))
for r in range(64):
    for g in range(64):
        for b in range(64):
            PxyInside = pdf([4 * r / 255, 4 * g / 255, 4 * b / 255], meanInside, covInside)
            PxyOutside = pdf([4 * r / 255, 4 * g / 255, 4 * b / 255], meanOutside, covOutside)
            Px = np.dot(PxyInside, Py1) + np.dot(PxyOutside, Py2)
            #some Px values get too small and round to zero, this debugs my div by zero bug
            if Px == 0:
                RGBTable[r,g,b] = 0
                continue
            RGBTable[r,g,b] =  BayesRule(PxyInside, Py1, Px)

#save my RGBTable in a folder
np.save(outputOutsideFolder + 'RGBTable', RGBTable)

#sanity check
print(RGBTable.shape)


