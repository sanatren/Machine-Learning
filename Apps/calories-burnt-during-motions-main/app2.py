import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Function to load models
def load_models():
    forest = pickle.load(open('random_forest_model.pkl', 'rb'))
    regressor = pickle.load(open('linear_regression_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return forest, regressor, scaler

# Function to predict calories burned
def predict_calories(model, scaler, input_data):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    return prediction

# Streamlit app
def main():
    st.title('Calories Burned Prediction App')

    # Sidebar with user input
    st.sidebar.header('User Input Features')

    # Example input data (replace with user inputs)
    example_data = {
        'Duration': 60,
        'Heart_Rate': 120,
        'Body_Temp': 98.6,
        'Gender': 'Male',  # Enter as 'Male' or 'Female'
        'Age': 30,
        'Height': 175,
        'Weight': 70,
    }

    # Display user input fields
    duration = st.sidebar.slider('Duration (minutes)', min_value=0, max_value=120, value=example_data['Duration'])
    heart_rate = st.sidebar.slider('Heart Rate', min_value=60, max_value=200, value=example_data['Heart_Rate'])
    body_temp = st.sidebar.slider('Body Temperature', min_value=95.0, max_value=105.0, value=example_data['Body_Temp'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'], index=0 if example_data['Gender'] == 'Male' else 1)
    age = st.sidebar.slider('Age', min_value=18, max_value=100, value=example_data['Age'])
    height = st.sidebar.slider('Height (cm)', min_value=140, max_value=220, value=example_data['Height'])
    weight = st.sidebar.slider('Weight (kg)', min_value=30, max_value=200, value=example_data['Weight'])

    # Convert gender to numeric for model prediction
    gender = 1 if gender == 'Female' else 0

    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp],
    })

    # Load models and scaler
    forest, regressor, scaler = load_models()

    # Make predictions
    forest_prediction = predict_calories(forest, scaler, input_data)
    regressor_prediction = predict_calories(regressor, scaler, input_data)

    # Display predictions
    st.header('Random Forest Regressor Prediction:')
    st.write(forest_prediction[0])

    st.header('Linear Regression Prediction:')
    st.write(regressor_prediction[0])

if __name__ == '__main__':
    main()