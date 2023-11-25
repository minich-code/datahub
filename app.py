import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np 
import pandas as pd 
import joblib

# Create app with Flask 
app = Flask(__name__)

# Load the pickle file 
linear_regression_model = joblib.load('linear_regression_model.pkl')
scalar = joblib.load('scalar.pkl')


# app.route 
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get JSON data from the request
        data = request.json['data']
        
        # Print and preprocess the input data
        print("Received Data:", data)
        new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))

        # Make predictions using the loaded model
        predictions = linear_regression_model.predict(new_data)
        print("Predictions:", predictions)

        # Convert predictions to a list (assuming 'predictions' is a NumPy array)
        output = predictions.tolist()

        return jsonify({'predictions': output})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)