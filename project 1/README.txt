<iframe width="100%" height="800" src="project 1 writeup.pdf">

READ ME

To run my code using my pre-trained model for test images, please add 1 folder named 'test' for all the test images. Then run predict.py to run on all the images in that folder.

There is one folder in the zip named labeledOutside which contains my trained model's lookup table for all possible RGB value probabilities, 'RGBTable.npy'.

When running predict.py, the terminal will print which file is being currently analyzed. Then the segmented test image will pop up in a new window, close that window to continue. Next the test image will pop up with a bounding box around the cone and a centroid dot, the terminal will print values for the distance and centroid coordinate. Close this test image window to move onto the next file.


************************************************************************************

If any folder names were changed or different from 'test', or you want to retrain on a new training set, please read the TODOs below.


************************************************************************************


Before running python files, please add a folder named 'test' for all test images and a folder named 'train' for all the training images, as well as an empty folder named 'labeledInside'. The folder 'labeledOutside' already exists.

There are 3 files that need to be run in this order:

1.  label.py
This will launch ROIpoly.

TODO IF FOLDER NAME IS NOT TRAIN/TEST: Change folder paths of trainPath, outputInsideFolder, and outputOutsideFolder accordingly. Currently outputInsideFolder is 'labeledInside/' and outputOutsideFolder is 'labeledOutside/'.

2.  createLookupTable.py
This will create a lookup table.

TODO IF FOLDER NAME IS NOT TRAIN/TEST: Make sure outputInsideFolder and outputOutsideFolder paths match that of label.py

3.  predict.py
This makes all the predictions.

TODO IF FOLDER NAME IS NOT TRAIN/TEST: Change testPath to have the correct path,  and make sure trainPath, outputInsideFolder, and outputOutsideFolder match that of label.py if they were changed.



DESCRIPTIONS OF FILES:

1.  label.py
This file will open up each training image so users can draw a boundary around each area of interest in the image (what you are trying to predict for the test images). The path of the training Images folder should be placed in trainPath, and 2 more folder paths should be defined for outputInsideFolder and outputOutsideFolder. These folders store the saved .npy files for the created masks, as well as a .npy of the img pixel values for those pixels inside and outside the mask, respectively. A .npy of the img pixels of the full img will also be stored in outputInsideFolder. The files will be titled 'maskInside.npy', 'insidePixels.npy', 'imgTotalPixels.npy', 'maskOutside.npy', and 'outsidePixels.npy'.

2.   createLookupTable.py
This file will create my lookup table of probability values for each possible RGB value. If the folder names have been changed, make sure to change the variables outputInsideFolder and outputOutsideFolder respectively. The values 'insidePixels.npy', 'outsidePixels.npy', 'imgTotalPixels.npy' will be loaded from these two folders. The only output from this file will be 'RGBTable.npy' which is saved in the outputOutsideFolder.

3.  predict.py
This file will display predictions and bounding boxes, and print out the predicted centroid coordinates as well as distance. Use the same trainPath and testPath as label.py, as well as outputInsideFolder and outputOutsideFolder paths. 'insidePixels.npy', 'outsidePixels.npy', 'imgTotalPixels.npy', and 'RGBTable.npy' will be loaded from the output folders.
