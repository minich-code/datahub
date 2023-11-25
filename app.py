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


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the request and convert it to a list of floats
    data = [float(x) for x in request.form.values()]

    # Transform the input data using the loaded scalar
    final_input = scalar.transform(np.array(data).reshape(1, -1))

    # Print the transformed input data (optional, for debugging)
    print(final_input)

    # Make a prediction using the loaded linear regression model
    output = linear_regression_model.predict(final_input)[0]

    # Render the prediction on the 'home.html' template
    return render_template("home.html", prediction_text="The house price prediction is {}".format(output))        
    

if __name__ == '__main__':
    app.run(debug=True)