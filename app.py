from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
from keras.models import model_from_json
from keras.models import model_from_json
import numpy
import os
import argparse
import base64

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

def load_model_sahana():
	# load json and create model
	json_file = open('model3.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("weights5.h5")
	print("Loaded model from disk")
	return loaded_model


def load_model():
    # Function to load and return neural network model
    json_file = open('models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights/model_A_weights.h5")
    return loaded_model

def create_img_sahana(path):
	img = cv2.imread(path, 0)
	print(img.shape)
	img = numpy.array(img)
	img = (img - 127.5) / 128
	print(img.shape[0])
	x_in = numpy.reshape(img, (1, img.shape[0], img.shape[1], 1))
	return x_in
def create_img(path):
    # Function to load,normalize and return image
    print(path)
    im = Image.open(path).convert('RGB')
    im = np.array(im)
    im = im / 255.0
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
    im = np.expand_dims(im, axis=0)
    return im

def predict(path):
    model = load_model()
    print('model is loaded')
    image = create_img(path)
    ans = model.predict(image)
    count = np.sum(ans)
    return count,image,ans
def predict_sahana(path):
	model = load_model_sahana()
	image= create_img_sahana()
	output = model.predict(image, batch_size=1, verbose=1, steps=None)
	return str(output[0])



# Define a flask app
app = Flask(__name__)

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predictResNet50', methods=['GET', 'POST'])
def predictResNet50():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)
        ans = predict(file_path)
        print(ans)
        #Print count, image, heat map
        #plt.imshow(img.reshape(img.shape[1],img.shape[2],img.shape[3]))
        #plt.show()
        #plt.imshow(hmap.reshape(hmap.shape[1],hmap.shape[2]) , cmap = c.jet )
        #plt.show()
        result = str(ans[0])
        return result
    return None


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
