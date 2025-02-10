from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file) # Loading the model

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file) # Loading the scaler

# Feature names required for the model
feature_names = ['CreditScore', 'Geography', 'Gender', 'Age', 
                 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
                 'IsActiveMember', 'EstimatedSalary']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract form data
            features = [float(request.form.get(name)) for name in feature_names]

            # Convert to DataFrame and apply scaling
            input_data = pd.DataFrame([features], columns=feature_names)
            scaled_features = scaler.transform(input_data)

            # Predict using the loaded model
            prediction = model.predict(scaled_features)[0]
            prediction = "Churn" if prediction == 1 else "No Churn"
        except Exception as e:
            prediction = f"Error: {e}"  # Debugging message

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
