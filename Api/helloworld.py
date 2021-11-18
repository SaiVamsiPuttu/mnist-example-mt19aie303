from flask import Flask
from flask import request
import sys
sys.path.insert(1, '/home/mt19aie303/MLOPS/mnist-example-mt19aie303/mnist-example-mt19aie303/mnist-example-mt19aie303')
import plot_digits_classification
import utils
import numpy as np
from skimage.transform import resize,rescale

app = Flask(__name__)

best_performingModel_path = "/home/mt19aie303/MLOPS/mnist-example-mt19aie303/mnist-example-mt19aie303/Models/model_gamma_0.001_94.81.pkl"
clf = utils.load_model(best_performingModel_path)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict",methods=["POST"])
def predict():
    input_json = request.json
    image = input_json['image']
    #print(image)
    #Image = np.array(image).resize(1,-1)
    imageArr = np.array(image)
    Image = imageArr.reshape((1,-1))
    predicted = clf.predict(Image)
    return "Given input image is of the digit: "+str(predicted[0])

