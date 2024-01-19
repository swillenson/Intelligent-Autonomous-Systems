import cv2
import matplotlib.pyplot as plt
import numpy as np
import roipoly
import os
from PIL import Image

#define all my personal folder paths, these can be changed if necessary
trainPath = "train/"
dir_list = os.listdir(trainPath)
outputInsideFolder = "labeledInside/"
outputOutsideFolder = "labeledOutside/"

#initialize some arrays to use in the train step for calculating probabilities
insidePixels = np.asarray([])
outsidePixels = np.asarray([])   
imgTotalPixels = np.asarray([])

#read each file within my train path and make a manual mask with roipoly
for file in dir_list[:]:
    img = plt.imread(trainPath + file)
    plt.imshow(img)
    my_roi = roipoly.RoiPoly(color = 'r')

    #maskInside is my mask for pixels within the roipoly boundary, mask Outside is the inverse of that
    maskInside = my_roi.get_mask(img[:,:,0])

    maskOutside = np.invert(maskInside)
    maskInside = np.asarray(maskInside)
    maskOutside = np.asarray(maskOutside)

    #imgInside is all pixel values within my box, imgOutside is all pixels outside my box
    imgInside = img[maskInside]
    imgOutside = img[maskOutside]

    #if else used here for empty array concat bug
    if len(insidePixels) == 0:
        insidePixels = imgInside
        outsidePixels = imgOutside
        imgTotalPixels = img
    else:
        insidePixels = np.concatenate((insidePixels, imgInside), axis = 0)
        outsidePixels = np.concatenate((outsidePixels, imgOutside), axis = 0)
        imgTotalPixels = np.concatenate((imgTotalPixels, img), axis = 0)

#calc mean of Trues and Falses
meanInside = np.mean(insidePixels, axis = 0)
meanOutside = np.mean(outsidePixels, axis = 0)

#sve files to specified folders from above to be used in train.py
np.save(outputInsideFolder + 'maskInside', maskInside)
np.save(outputInsideFolder + 'insidePixels', insidePixels)
np.save(outputInsideFolder + 'imgTotalPixels', imgTotalPixels)

np.save(outputOutsideFolder + 'maskOutside', maskOutside)
np.save(outputOutsideFolder + 'outsidePixels', outsidePixels)





# print("in:", meanInside, "out:", meanOutside)