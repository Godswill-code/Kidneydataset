import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
with open('kidney_ml_model1.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the list of 20 input features
input_features = [
    'rbc', 'pc', 'pcc', 'ba', 'rc', 'wc', 'pcv', 'htn', 'dm', 'cad', 
    'appet', 'pe', 'ane', 'age', 'bgr', 'bu', 'sod', 'pot', 'hemo', 'sg'
]

# Define Streamlit app
def main():
    st.title("CKD Prediction")

    # User input for the 20 features
    input_data = []
    for feature in input_features:
        value = st.number_input(f"Enter value for {feature}", value=0.0)
        input_data.append(value)

    # Convert input to DataFrame
    input_data = pd.DataFrame([input_data], columns=input_features)

    # Add dummy columns if needed
    expected_features = 421  # Total features expected by scaler and model
    current_features = input_data.shape[1]

    if current_features < expected_features:
        for i in range(expected_features - current_features):
            input_data[f"dummy_feature_{i}"] = 0

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display result
    st.write("Prediction:", "CKD Detected" if prediction[0] == 1 else "No CKD Detected")

if __name__ == "__main__":
    main()
