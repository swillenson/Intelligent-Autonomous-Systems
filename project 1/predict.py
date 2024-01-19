import cv2
import matplotlib.pyplot as plt
import numpy as np
import roipoly
import os
from PIL import Image
from skimage import data, util
from skimage.measure import label, regionprops
import math

#define my folder paths, same as label.py
trainPath = "train/"
testPath = "test/"

#change this path variable depending on whether you're loading from train/testPath
path = testPath
dir_list = os.listdir(path)

outputInsideFolder = "labeledInside/"
outputOutsideFolder = "labeledOutside/"

#load my lookup table of all possible PDF values
RGBTable = np.load(outputOutsideFolder + 'RGBTable.npy')

#initialize arrays to be filled in while looping through each img file
probAllImgs = np.asarray([])
centroids = []
distances = []

# calculate focal length knowing its the same camera for my train images and test set to come later
def getFocalLength(maxr, minr, maxc, minc):
    v = maxr - minr
    u = maxc - minc
    coneW = 7.5
    coneH = 17
    #v is found from previous training run's bounding box, knowing that maxr = 490 & minr = 332 -> v = 158
    f = v * 6 / 17
    return (17 * f / v)

#set threshold to be used for predictions
threshold = 0.99999999999

#loop through test images, calculating probabilities from Bayes Rule and use these to 
#predict whether or not each pixel from my test images are Cones or Not
#load probabilities from my lookup table
#initialize centroids & distances lists in case of multiple cones/bounding boxes
for file in dir_list[:]:
    print("analyzing file: ", file)
    centroids = []
    distances = []
    img = plt.imread(path + file)
    probImg = np.zeros((img.shape[0], img.shape[1]))
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            probPix = RGBTable[int(255 * img[i,j][0] / 4), int(255 * img[i,j][1] / 4), int(255 * img[i,j][2] / 4)]
            probImg[i,j] = probPix > threshold
    plt.imshow(probImg)
    plt.show()


    #initialize values for bounding box
    label_img = label(probImg)
    regions = regionprops(label_img)
    fig, ax = plt.subplots()
    ax.imshow(img)

    #draw bounding box
    for props in regions:
        y0, x0 = props.centroid

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        #I calculated the focal length manually knowing that the camera is the same for train and test images
        #using the pinhole camera formula I can find Z using this f for each img file
        #55.7647 is my found value of my focal length from a previous train run
        #17 is the height of the cone
        distance = 17 * (55.7647) / (maxr - minr)
        if distance > 100:
            continue

        ax.plot(x0, y0, '.g', markersize=5)
        ax.plot(bx, by, '-b', linewidth=2.5)

        distances.append(distance)
        centroids.append([x0, y0])


    #print my final calculations for distance and centroid
    print("distance for ", file, " is: ", distances)
    print("centroid for ", file, " is: ", centroids)
    plt.show()

