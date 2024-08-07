import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE_PATH = "penguin_random_forest_model.pkl"

# Function to load data
@st.cache
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    X_raw = df.drop('species', axis=1)
    y_raw = df.species
    return df, X_raw, y_raw

# Function to load or train model
@st.cache
def get_model():
    if os.path.exists(MODEL_FILE_PATH):
        model = joblib.load(MODEL_FILE_PATH)
    else:
        model = RandomForestClassifier(n_estimators=100)
        df, X_raw, y_raw = load_data()
        model.fit(X_raw, y_raw)
        joblib.dump(model, MODEL_FILE_PATH)
    return model

# Function to get user input
def get_user_input():
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))
    
    data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    features = pd.DataFrame(data, index=[0])
    return features

st.title('🤖 Machine Learning App')

st.info(
    '''
    This is an app that builds a machine learning model in Python with Streamlit!
    \n Source: 
    https://github.com/dataprofessor/dp-machinelearning/blob/master/streamlit_app.py
    https://www.youtube.com/@streamlitofficial
    '''
)

# Load data
df, X_raw, y_raw = load_data()

with st.expander('Data'):
    st.write('**Raw data**')
    st.write(df)

    st.write('**X**')
    st.write(X_raw)

    st.write('**y**')
    st.write(y_raw)

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Initialize the session state to store the input data
if 'input_df' not in st.session_state:
    st.session_state['input_df'] = pd.DataFrame()
    input_df = st.session_state['input_df']

if 'input_penguins' not in st.session_state:
    st.session_state['input_penguins'] = pd.DataFrame()
    input_penguins = st.session_state['input_penguins']

with st.sidebar:
    st.header('Input features')
    input_df = get_user_input()

# Load or train model
model = get_model()

# Make predictions
if st.sidebar.button('Predict'):
    prediction = model.predict(input_df)
    st.sidebar.write(f'Prediction: {prediction[0]}')

