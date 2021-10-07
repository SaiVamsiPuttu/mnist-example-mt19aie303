import math
import sys
sys.path.insert(1, '/tmp/MLOPS/mnist-example-mt19aie303/mninst-example-mt19aie303')
import plot_digits_classification
import utils
import sklearn.datasets as datasets
import os
from sklearn import metrics
import pickle


def test_model_writing():
    testSplitRatio = 0.15
    valSplitRatio = 0.15
    digits = datasets.load_digits()
    gammaValues = [0.001]
    savedModelFolderPath = "/tmp/MLOPS/mnist-example-mt19aie303/Models/"
    preProcessedData = utils.preProcess(8,digits)
    X_train,X_test,X_val,y_train,y_test,y_val = utils.create_splits(preProcessedData,digits,testSplitRatio,valSplitRatio)
    metricsDf = utils.training(X_train,X_val,y_train,y_val,gammaValues,savedModelFolderPath)
    #print("Model Path: ",metricsDf['Path to Model'][0])
    modelName = metricsDf['Path to Model'][0]
    assert os.path.isfile('/tmp/MLOPS/mnist-example-mt19aie303/Models/'+ str(modelName)+'.pkl')


def test_small_data_overfit_checking():
    testSplitRatio = 0.15
    valSplitRatio = 0.15
    digits = datasets.load_digits()
    gammaValues = [0.001]
    savedModelFolderPath = "/tmp/MLOPS/mnist-example-mt19aie303/Models/"
    preProcessedData = utils.preProcess(8,digits)
    X_train,X_test,X_val,y_train,y_test,y_val = utils.create_splits(preProcessedData,digits,testSplitRatio,valSplitRatio)
    metricsDf = utils.training(X_train[:100],X_train[:100],y_train[:100],y_train[:100],gammaValues,savedModelFolderPath)
    modelName = metricsDf['Path to Model'][0]
    model_path = '/tmp/MLOPS/mnist-example-mt19aie303/Models/'+ str(modelName)+'.pkl'

    with open(model_path,'rb') as f:
            model = pickle.load(f)
    
    y_train_predicted = model.predict(X_train[:100])
    y_test_predicted = model.predict(X_test)
    train_acc = round(100*metrics.accuracy_score(y_train[:100],y_train_predicted),2)
    test_acc = round(100*metrics.accuracy_score(y_test,y_test_predicted),2)
    f1_score_test = round(100*metrics.f1_score(y_test,y_test_predicted,average='weighted'))
    f1_score_train = round(100*metrics.f1_score(y_train[:100],y_train_predicted,average='weighted'))
    assert (train_acc > test_acc + 5 and f1_score_train > f1_score_test + 0.3)
