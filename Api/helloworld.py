from flask import Flask
from flask import request
import sys
sys.path.insert(1, '/home/mt19aie303/MLOPS/mnist-example-mt19aie303/mnist-example-mt19aie303/mnist-example-mt19aie303')
import plot_digits_classification
import utils
import numpy as np
from skimage.transform import resize,rescale

app = Flask(__name__)

best_performingModel_path_svm = "/home/mt19aie303/MLOPS/mnist-example-mt19aie303/mnist-example-mt19aie303/Models/model_gamma_0.001_94.81.pkl"
clf = utils.load_model(best_performingModel_path_svm)

best_performingModel_path_dt= "/home/mt19aie303/MLOPS/mnist-example-mt19aie303/mnist-example-mt19aie303/Models/DTmodel_gamma_2_78.52.pkl"
clfDT = utils.load_model(best_performingModel_path_dt)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/svm_predict",methods=["POST"])
def svm_predict():
    input_json = request.json
    image = input_json['image']
    #print(image)
    #Image = np.array(image).resize(1,-1)
    imageArr = np.array(image)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    return "Given input image is of the digit: "+str(predicted[0])

@app.route("/decision_tree_predict",methods=["POST"])
def decision_tree_predict():
    input_json = request.json
    image = input_json['image']
    #print(image)
    #Image = np.array(image).resize(1,-1)
    imageArr = np.array(image)
    Image = imageArr.reshape((1,-1))
    predicted = clfDT.predict(Image)
    return "Given input image is of the digit: "+str(predicted[0])



