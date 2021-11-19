import math
import sys
sys.path.insert(1, '/home/mt19aie303/MLOPS/mnist-example-mt19aie303/mnist-example-mt19aie303/mnist-example-mt19aie303')
import plot_digits_classification
import utils
import sklearn.datasets as datasets
import os
from sklearn import metrics
import pickle
import numpy as np
from sklearn.metrics import accuracy_score,precision_score
from sklearn.model_selection import train_test_split

best_performingModel_path_SVM = "/home/mt19aie303/MLOPS/mnist-example-mt19aie303/mnist-example-mt19aie303/Models/model_gamma_0.001_94.81.pkl"
best_performingModel_path_DT = "/home/mt19aie303/MLOPS/mnist-example-mt19aie303/mnist-example-mt19aie303/Models/DTmodel_gamma_0.2_78.52.pkl"
clf = utils.load_model(best_performingModel_path_SVM)
clf_DT = utils.load_model(best_performingModel_path_DT)
data = datasets.load_digits()

preProcessedData = utils.preProcess(8,data)

X_train, X_test, y_train, y_test = train_test_split(preProcessedData, data.target, train_size=(1-(0.15+0.15)), shuffle=False)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=float(0.15/(0.15+0.15)), shuffle=False)


def test_digit_correct_0_svm():
    image_0 = data.images[list(data.target).index(0)]
    imageArr = np.array(image_0)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 0)

def test_digit_correct_1_svm():
    image_1 = data.images[list(data.target).index(1)]
    imageArr = np.array(image_1)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 1)

def test_digit_correct_2_svm():
    image_2 = data.images[list(data.target).index(2)]
    imageArr = np.array(image_2)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 2)

def test_digit_correct_3_svm():
    image_3 = data.images[list(data.target).index(3)]
    imageArr = np.array(image_3)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 3)

def test_digit_correct_4_svm():
    image_4 = data.images[list(data.target).index(4)]
    imageArr = np.array(image_4)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 4)

def test_digit_correct_5_svm():
    #image_5 = data.images[list(data.target).index(5)]
    image_5 = data.images[15]
    imageArr = np.array(image_5)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 5)

def test_digit_correct_6_svm():
    image_6 = data.images[list(data.target).index(6)]
    imageArr = np.array(image_6)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 6)

def test_digit_correct_7_svm():
    image_7 = data.images[list(data.target).index(7)]
    imageArr = np.array(image_7)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 7)

def test_digit_correct_8_svm():
    image_8 = data.images[list(data.target).index(8)]
    imageArr = np.array(image_8)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 8)

def test_digit_correct_9_svm():
    image_9 = data.images[list(data.target).index(9)]
    imageArr = np.array(image_9)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    assert (predicted[0] == 9)

def test_digit_correct_0_DT():
    image_0 = data.images[list(data.target).index(0)]
    imageArr = np.array(image_0)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 0)

def test_digit_correct_1_DT():
    image_1 = data.images[list(data.target).index(1)]
    imageArr = np.array(image_1)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 1)

def test_digit_correct_2_DT():
    image_2 = data.images[list(data.target).index(2)]
    imageArr = np.array(image_2)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 2)

def test_digit_correct_3_DT():
    image_3 = data.images[list(data.target).index(3)]
    #image_3 = data.images[3]
    imageArr = np.array(image_3)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 3)

def test_digit_correct_4_DT():
    image_4 = data.images[list(data.target).index(4)]
    imageArr = np.array(image_4)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 4)

def test_digit_correct_5_DT():
    image_5 = data.images[list(data.target).index(5)]
    imageArr = np.array(image_5)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 5)

def test_digit_correct_6_DT():
    image_6 = data.images[list(data.target).index(6)]
    imageArr = np.array(image_6)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 6)

def test_digit_correct_7_DT():
    image_7 = data.images[list(data.target).index(7)]
    imageArr = np.array(image_7)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 7)

def test_digit_correct_8_DT():
    image_8 = data.images[list(data.target).index(8)]
    imageArr = np.array(image_8)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 8)

def test_digit_correct_9_DT():
    image_9 = data.images[list(data.target).index(9)]
    imageArr = np.array(image_9)
    Image = imageArr.reshape((1,-1))
    predicted = clf_DT.predict(Image)
    assert (predicted[0] == 9)

def test_check_classAcc_SVM():

    preProcessedData = utils.preProcess(8,data)

    X_train, X_test, y_train, y_test = train_test_split(preProcessedData, data.target, train_size=(1-(0.15+0.15)), shuffle=False)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=float(0.15/(0.15+0.15)), shuffle=False)

    for digit in range(10):
        label_indices = np.where(y_val==digit)[0]
        X_val_label = list(map(X_val.__getitem__, label_indices))
        pred = clf.predict(X_val_label)
        valAcc = accuracy_score(pred, y_val[label_indices])
        assert valAcc > 0.35

def test_check_classAcc_DT():

    for digit in range(10):
        label_indices = np.where(y_val==digit)[0]
        X_val_label = list(map(X_val.__getitem__, label_indices))
        pred = clf_DT.predict(X_val_label)
        valAcc = accuracy_score(pred, y_val[label_indices])
        assert valAcc > 0.35

    
