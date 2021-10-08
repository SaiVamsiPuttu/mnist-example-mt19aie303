import math
import sys
sys.path.insert(1, '/tmp/MLOPS/mnist-example-mt19aie303/mninst-example-mt19aie303')
import plot_digits_classification
import utils
import sklearn.datasets as datasets
import os
from sklearn import metrics
import pickle




def test_create_split_for100Samples():
    testSplitRatio = 0.20
    valSplitRatio = 0.10
    digits = datasets.load_digits()
    n = 100
    preProcessedData = utils.preProcess(8,digits)
    X_train,X_test,X_val,y_train,y_test,y_val = utils.create_splits(preProcessedData,digits.images[:n],digits.target[:n],testSplitRatio,valSplitRatio)
    assert (len(X_train) == (n - int(round(((testSplitRatio+valSplitRatio) * n),0)))) and (len(X_test)-1 == int(round((testSplitRatio * n),0))) and (len(X_val) == int(round((valSplitRatio * n),0)) -1) and (len(X_train) + len(X_test) + len(X_val) == n)


def test_create_split_for9Samples():
    testSplitRatio = 0.20
    valSplitRatio = 0.10
    digits = datasets.load_digits()
    n = 9
    preProcessedData = utils.preProcess(8,digits)
    X_train,X_test,X_val,y_train,y_test,y_val = utils.create_splits(preProcessedData,digits.images[:n],digits.target[:n],testSplitRatio,valSplitRatio)
    assert (len(X_train) == (n - int(round(((testSplitRatio+valSplitRatio) * n),0)))) and (len(X_test) == int(round((testSplitRatio * n),0))) and (len(X_val) == int(round(((valSplitRatio) * n),0))) and (len(X_train) + len(X_test) + len(X_val) == n)