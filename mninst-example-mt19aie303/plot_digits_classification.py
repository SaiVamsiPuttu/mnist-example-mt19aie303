"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)


# Standard scientific Python imports
import matplotlib.pyplot as plt
import sys, getopt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize,rescale
import numpy as np
import pandas as pd
import pickle
import os
import utils

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.





"""_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)"""

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images


def main(argv):

        digits = datasets.load_digits()

        n = len(sys.argv[1])
        gammaValues = sys.argv[1][1:n-1]
        gammaValues = gammaValues.split(',')
        testSplitRatio = float(sys.argv[2])
        valSplitRatio = float(sys.argv[3])
        savedModelFolderPath = sys.argv[4]
        preProcessedData = utils.preProcess(8,digits)
        X_train,X_test,X_val,y_train,y_test,y_val = utils.create_splits(preProcessedData,digits,testSplitRatio,valSplitRatio)
        metricsDf = utils.training(X_train,X_val,y_train,y_val,gammaValues,savedModelFolderPath)
        
        utils.testing(metricsDf,X_test,y_test,savedModelFolderPath)
 

if __name__ == "__main__":
        main(sys.argv[1:])
 



