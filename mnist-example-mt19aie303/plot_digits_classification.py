"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize,rescale
import numpy as np
import pandas as pd

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

digits = datasets.load_digits()
metrics_table = np.array([[0.000,0,0]])


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

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
n_samples = len(digits.images)
imageSize = [16,32,64]
splitratio = [10,20,30]
i = 0

#for size in imageSize:
for gammaParam in [0.001,0.01,0.05,0.1,0.25,0.5,0.75,1,10,100]:

        #print("#####################--------------------Evaluating model performance on Images of Size 8*8 with gamma value of {0} ##################---------------------".format(gammaParam))

        #resizedImgs = np.zeros((len(digits.images),size,size))

        #resizedImgs = np.array(map(lambda img : resize(img,(size,size)),digits.images))
        """i = 0
        for img in digits.images:
                resizedImgs[i] = resize(img,(size,size))
                i += 1 """
        #data = resizedImgs.reshape((n_samples,-1))
        data = digits.images.reshape((n_samples,-1))

        # Create a classifier: a support vector classifier
        clf = svm.SVC(gamma=gammaParam)

        # Split data into 10% train and 50% test subsets
        X_train, X_test, y_train, y_test = train_test_split(data, digits.target, train_size=0.7, shuffle=False)

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)
        predicted_val = clf.predict(X_val)

        ###############################################################################
        # Below we visualize the first 4 test samples and show their predicted
        # digit value in the title.

        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, prediction in zip(axes, X_test, predicted):
                ax.set_axis_off()
                image = image.reshape(8, 8)
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                ax.set_title(f'Prediction: {prediction}')

        ###############################################################################
        # :func:`~sklearn.metrics.classification_report` builds a text report showing
        # the main classification metrics.
        #print()
        #print(f"Classification report for classifier {clf}:\n"f"{metrics.classification_report(y_test, predicted)}\n")

        ###############################################################################
        # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
        # true digit values and the predicted digit values.
        #print()
        disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        #print(f"Confusion matrix:\n{disp.confusion_matrix}")
        #print("Accuracy on train Data: ",metrics.accuracy_score(y_train,predicted))
        #print("Accuracy on test Data: ",metrics.accuracy_score(y_test,predicted))
        #print("Accuracy on val Data: ",metrics.accuracy_score(y_val,predicted_val))
        metrics_table = np.insert(metrics_table,i,[gammaParam,100*metrics.accuracy_score(y_test,predicted),100*metrics.accuracy_score(y_val,predicted_val)],axis=0)
        #print(metrics_table)
        #plt.show()
        #print()
        i += 1
#metrics_table
metrics_table = pd.DataFrame(metrics_table,columns=['Gamma Value','Accuracy on TEST Data','Accuracy on Validation Data'])
print(metrics_table)
