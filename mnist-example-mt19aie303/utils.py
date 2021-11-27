# Standard scientific Python imports
import matplotlib.pyplot as plt
import sys, getopt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics,tree
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

def training(X_train,X_val,X_test,y_train,y_val,y_test,gammaValues,hypPind):
        metrics_table = np.array([['None',0,0.0,0,0,0,0,0,0,0,0,0,'None']])

        i = 0
        #gammaParam = float(gammaParam)
        clf = tree.DecisionTreeClassifier(max_features=gammaValues[0],min_samples_split=int(gammaValues[1]),min_weight_fraction_leaf=float(gammaValues[2]))

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)
        predicted_train = clf.predict(X_train)
        predicted_val = clf.predict(X_val)
        predicted_test = clf.predict(X_test)

        val_acc = round(100*metrics.accuracy_score(y_val,predicted_val),2)
        train_acc = round(100*metrics.accuracy_score(y_train,predicted_train),2)
        test_acc = round(100*metrics.accuracy_score(y_test,predicted_test),2)

        #save_path = "/tmp/MLOPS/mnist-example-mt19aie303/Models/"
        #save_path = modelPath

        #name_of_file = save_path+"model_gamma_{0}_{1}".format(gammaParam,val_acc)

        #completeName = os.path.join(save_path, name_of_file+".pkl")
        print("gammaValues[0]: ",gammaValues[0])
        metrics_table = np.insert(metrics_table,0,[str(gammaValues[0]),int(gammaValues[1]),float(gammaValues[2]),train_acc,val_acc,test_acc,0,0,0,0,0,0,'None'],axis=0)        

        """with open(completeName, 'wb') as f:
                pickle.dump(clf, f)"""
        
        metrics_table = pd.DataFrame(metrics_table,columns=['max_features','min_samples_leaf','min_weight_fraction_leaf','Train_Run1','Dev_Run1','Test_Run1','Train_Run2','Dev_Run2','Test_Run2','Train_Run3','Dev_Run3','Test_Run3','Observation'])

        clf.fit(X_train, y_train)
        predicted_train = clf.predict(X_train)
        predicted_val = clf.predict(X_val)
        predicted_test = clf.predict(X_test)
        val_acc = round(100*metrics.accuracy_score(y_val,predicted_val),2)
        train_acc = round(100*metrics.accuracy_score(y_train,predicted_train),2)
        test_acc = round(100*metrics.accuracy_score(y_test,predicted_test),2)
        metrics_table['Train_Run2'] = train_acc
        metrics_table['Dev_Run2'] = val_acc
        metrics_table['Test_Run2'] = test_acc

        clf.fit(X_train, y_train)
        predicted_train = clf.predict(X_train)
        predicted_val = clf.predict(X_val)
        predicted_test = clf.predict(X_test)
        val_acc = round(100*metrics.accuracy_score(y_val,predicted_val),2)
        train_acc = round(100*metrics.accuracy_score(y_train,predicted_train),2)
        test_acc = round(100*metrics.accuracy_score(y_test,predicted_test),2)
        metrics_table['Train_Run3'] = train_acc
        metrics_table['Dev_Run3'] = val_acc
        metrics_table['Test_Run3'] = test_acc

        """if(hypPind != 1):
                df_existing = pd.read_csv("/home/mt19aie303/MLOPS/mnist-example-mt19aie303/mnist-example-mt19aie303/results.xlsx")
                df_existing.append(metrics_table, ignore_index = True)"""

        return metrics_table

 

"""def testing(metrics_table,X_test,y_test,model_path):

        
        best_gamma_value = float(metrics_table[metrics_table['Accuracy on Validation Data'] == metrics_table['Accuracy on Validation Data'].max()]['Gamma Value'])
        print("Best performing gamma value is {0} ".format(best_gamma_value))

        model_path = model_path + str(metrics_table[metrics_table['Accuracy on Validation Data'] == metrics_table['Accuracy on Validation Data'].max()]['Path to Model'][1])+'.pkl'

        with open(model_path,'rb') as f:
            model = pickle.load(f)

        predicted_testSet = model.predict(X_test)
        print("Accuracy on test data with best gamma value is {0}".format(100*metrics.accuracy_score(y_test,predicted_testSet)))
"""