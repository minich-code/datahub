import pickle
from flask import Flask, request, app, jasonify, url_for, render_template
import numpy as np 
import pandas as pd 


app = Flask(__name__)
# Load the model 
linear_regression_model = pickle.load('linear_regression_model.pkl', 'rb')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
