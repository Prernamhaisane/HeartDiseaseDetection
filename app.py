import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
# Load the saved model
@st.cache_data
def load_model():
    current_directory = os.getcwd()
    model_file_path = os.path.join(current_directory, 'heartdisease.h5')
    xgb_model = joblib.load(model_file_path)
    return xgb_model


def main():
    st.title('Heart Disease Prediction')
    # Load the model
    model = load_model()

    # Define options for Chest Pain Type
    cp_options = {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-Anginal Pain",
        3: "Asymptomatic"
    }

    # Define options for Sex
    sex_options = {
        1: "Male",
        0: "Female"
    }

    # Define options for Fasting Blood Sugar
    fbs_options = {
        1: "No (> 120 mg/dl)",
        0: "Yes (<= 120 mg/dl)"
    }

    # Define options for Exercise Induced Angina
    exang_options = {
        1: "Yes",
        0: "No"
    }

    # Define options for Thalassemia
    thal_options = {
        0: "Normal",
        1: "Fixed Defect",
        2: "Reversible Defect",
        3: "Unknown"
    }

    # Prediction
    st.subheader('Make Predictions')
    age = st.number_input('Age')
    sex = st.selectbox('Sex', list(sex_options.values()))
    cp = st.selectbox('Chest Pain Type', list(cp_options.values()))
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)')
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', list(fbs_options.values()))
    thalach = st.number_input('Maximum Heart Rate Achieved')
    exang = st.selectbox('Exercise Induced Angina', list(exang_options.values()))
    thal = st.selectbox('Thalassemia', list(thal_options.values()))

    if st.button('Predict'):
        input_data = np.array([age, list(sex_options.keys())[list(sex_options.values()).index(sex)], list(cp_options.keys())[list(cp_options.values()).index(cp)], trestbps, list(fbs_options.keys())[list(fbs_options.values()).index(fbs)], thalach, list(exang_options.keys())[list(exang_options.values()).index(exang)], list(thal_options.keys())[list(thal_options.values()).index(thal)]])  
        input_data = input_data.reshape(1, -1)
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.write('The person does not have heart disease.')
        else:
            st.write('The person is suffering from heart disease.')

if __name__ == '__main__':
    main()
