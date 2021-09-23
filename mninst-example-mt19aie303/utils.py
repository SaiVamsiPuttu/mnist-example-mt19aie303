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

def training(X_train,X_val,y_train,y_val,gammaValues,modelPath):

        metrics_table = np.array([[0.000,0,"Sample_Path"]])

        for gammaParam in gammaValues:
            i = 0
            gammaParam = float(gammaParam)
            clf = svm.SVC(gamma=gammaParam)


            # Learn the digits on the train subset
            clf.fit(X_train, y_train)

            predicted_val = clf.predict(X_val)

            val_acc = round(100*metrics.accuracy_score(y_val,predicted_val),2)

            if val_acc > 10:

                    #save_path = "/tmp/MLOPS/mnist-example-mt19aie303/Models/"
                    save_path = modelPath

                    name_of_file = save_path+"model_gamma_{0}_{1}".format(gammaParam,val_acc)

                    completeName = os.path.join(save_path, name_of_file+".pkl")

                    metrics_table = np.insert(metrics_table,i,[gammaParam,val_acc,"model_gamma_{0}_{1}".format(gammaParam,val_acc)],axis=0)        

                    with open(completeName, 'wb') as f:
                            pickle.dump(clf, f)

                    i += 1
        
        metrics_table = pd.DataFrame(metrics_table,columns=['Gamma Value','Accuracy on Validation Data','Path to Model'])

        return metrics_table

 

def testing(metrics_table,X_test,y_test,model_path):

        
        best_gamma_value = float(metrics_table[metrics_table['Accuracy on Validation Data'] == metrics_table['Accuracy on Validation Data'].max()]['Gamma Value'])
        print("Best performing gamma value is {0} ".format(best_gamma_value))

        model_path = model_path + str(metrics_table[metrics_table['Accuracy on Validation Data'] == metrics_table['Accuracy on Validation Data'].max()]['Path to Model'][1])+'.pkl'

        with open(model_path,'rb') as f:
            model = pickle.load(f)

        predicted_testSet = model.predict(X_test)
        print("Accuracy on test data with best gamma value is {0}".format(100*metrics.accuracy_score(y_test,predicted_testSet)))
