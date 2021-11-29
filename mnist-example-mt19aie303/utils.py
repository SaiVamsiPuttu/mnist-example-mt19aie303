# Standard scientific Python imports
import matplotlib.pyplot as plt
import sys, getopt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize,rescale
import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns







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


def preProcess(resizeParam,data):
        n_samples = len(data.images)
        i = 0
        if resizeParam != 8:
            resizedImgs = np.zeros((len(data.images),resizeParam,resizeParam))
            for img in data.images:
                    resizedImgs[i] = resize(img,(resizeParam,resizeParam))
                    i += 1 

            processedData = resizedImgs.reshape((n_samples,-1))
        else:
            processedData = data.images.reshape((n_samples, -1))

        return processedData

# Split data into 70% train and 30% test subsets
def create_splits(data,digits,test_size,valid_size):

        X_train, X_test, y_train, y_test = train_test_split(data, digits.target, train_size=(1-(test_size+valid_size)), shuffle=False)

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=float(test_size/(test_size+valid_size)), shuffle=False)

        return X_train,X_test,X_val,y_train,y_test,y_val

def training(X_train,X_test,X_val,y_train,y_test,y_val):
        #print("hyperParams: ",hyperParams[0])
        metrics_table = np.array([[0.000,0,"Sample_Path"]])
        metrics_tableDT = np.array([[0.000,0,"Sample_Path"]])

        trainDataPercnt = []
        f1Score = []
        roc_scores = {}
        metricsAPRF = {}
        for percnt in range(10,110,10):

                percntXtrain = X_train[0:int((len(X_train)*(percnt/100)))]
                percntYtrain = y_train[0:int((len(X_train)*(percnt/100)))]
                # Learn the digits on the train subset
                valAcc = []
                for gammaval in ([0.0008,0.01,0.001]):
                        clf = svm.SVC(probability=True,gamma=gammaval)
                        clf.fit(percntXtrain, percntYtrain)
                        predicted_val = clf.predict(X_val)
                        valAcc.append(round(100*metrics.accuracy_score(y_val,predicted_val),2))
                optGammaVal = valAcc.index(max(valAcc))
                clf = svm.SVC(probability=True,gamma=[0.0008,0.01,0.001][optGammaVal])
                clf.fit(percntXtrain, percntYtrain)
                predicted_y = clf.predict(X_test)
                f1ScoreInternal = round(100*metrics.f1_score(y_test, predicted_y, average='macro'),2)
                trainDataPercnt.append(percnt)
                f1Score.append(f1ScoreInternal)
                roc_scores[percnt] = round(100*metrics.roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr'),2)


        return trainDataPercnt,f1Score,roc_scores

 

def testing(metrics_table,X_test,y_test,model_path):

        print("---------------- Performance Table ----------------------")
        print(metrics_table)
        print()
        best_gamma_value = float(metrics_table[metrics_table['Accuracy on Validation Data'] == metrics_table['Accuracy on Validation Data'].max()]['Param Value'].values[0])
        print("Best performing gamma value is {0} ".format(best_gamma_value))

        model_path = model_path + str(metrics_table[metrics_table['Accuracy on Validation Data'] == metrics_table['Accuracy on Validation Data'].max()]['Path to Model'].values[0])+'.pkl'

        with open(model_path,'rb') as f:
            model = pickle.load(f)

        predicted_testSet = model.predict(X_test)
        print("Accuracy on test data with best gamma value is {0}".format(100*metrics.accuracy_score(y_test,predicted_testSet)))


def load_model(model_path):
        
        with open(model_path,'rb') as f:
            model = pickle.load(f)
        
        return model