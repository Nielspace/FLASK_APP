import warnings
from flask import Flask, render_template, request
import cv2
import numpy as np
import keras.models
import re
import base64
import sys 
import os
from model.load import * 

warnings.filterwarnings("ignore")


#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))
	

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    parseImage(request.get_data())
    x = cv2.imread('output.png')
    x = np.invert(x)
    x = cv2.resize(x,(28,28))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    x = x.reshape(1,28,28,1)
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        return response 


def parseImage(imgData):
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.debug = True
 