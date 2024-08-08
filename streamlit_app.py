import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# File path for saving the trained model
MODEL_FILE_PATH = "penguin_random_forest_model.pkl"

# Title of the Streamlit app
st.title('🤖 Machine Learning App to Predict Penguins')

# Display information about the app
st.info(
    '''
    This app builds a machine learning model in Python with Streamlit!
    \n Source: 
    https://github.com/dataprofessor/dp-machinelearning/blob/master/streamlit_app.py
    https://www.youtube.com/@streamlitofficial
    '''
)

# Function to load data from a CSV file
@st.cache
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    return df

# Function to load or train the machine learning model
@st.cache
def get_model(X1, y1):
    # Check if model file exists
    if os.path.exists(MODEL_FILE_PATH):
        # Load the model from file
        model = joblib.load(MODEL_FILE_PATH)
        st.success("Model loaded from file.")
    else:
        # Train the model and save it to file
        model = RandomForestClassifier()
        model.fit(X1, y1)
        joblib.dump(model, MODEL_FILE_PATH)
        st.success("Model trained and saved to file.")
    return model

# Function to get user input for prediction
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
    return pd.DataFrame(data, index=[0])

# Function to encode categorical features
def encode_features(X_raw, y_raw, input_df):
    # Encoding islands
    islands = ['Biscoe', 'Dream', 'Torgersen']
    island_encoder = {island: i for i, island in enumerate(islands)}
    X_raw['island'] = X_raw['island'].map(island_encoder)
    input_df['island'] = input_df['island'].map(island_encoder)

    # Encoding sex
    genders = ['male', 'female']
    gender_encoder = {gender: i for i, gender in enumerate(genders)}
    X_raw['sex'] = X_raw['sex'].map(gender_encoder)
    input_df['sex'] = input_df['sex'].map(gender_encoder)

    # Encoding species
    species = ['Adelie', 'Chinstrap', 'Gentoo']
    species_encoder = {species: i for i, species in enumerate(species)}
    y_raw = y_raw.map(species_encoder)

    return X_raw, y_raw, input_df

# Function to make predictions and display the results
def predict_penguin(model, input_df):
    prediction_proba = model.predict_proba(input_df)
    prediction = model.predict(input_df)
    df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])
    df_prediction_proba['Predicted_Species'] = prediction

    # Display prediction probabilities and predicted species
    st.subheader('Prediction Probabilities and Predicted Species')
    st.dataframe(df_prediction_proba, column_config={
        'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', min_value=0, max_value=1),
        'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', min_value=0, max_value=1),
        'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', min_value=0, max_value=1),
        'Predicted_Species': st.column_config.TextColumn('Predicted Species', width='medium')
    }, hide_index=False)

# Initialize the session state to store the input data
if 'input_df' not in st.session_state:
    st.session_state['input_df'] = pd.DataFrame()
    input_df = st.session_state['input_df']
if 'input_penguins' not in st.session_state:
    st.session_state['input_penguins'] = pd.DataFrame()
    input_penguins = st.session_state['input_penguins']

# Show raw data
with st.expander('Data'):
    st.write('**Raw data**')
    df = load_data()

    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.write('**y**')
    y_raw = df['species']
    st.dataframe(y_raw)

# Data visualization
with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input features to predict
with st.sidebar:
    st.header('Input features')
    input_df = get_user_input()

# Show input features
with st.expander('Input features'):
    st.write('**Input penguin**')
    st.dataframe(input_df)

# Encode features
X, y, input_df = encode_features(X_raw, y_raw, input_df)

# Show encoded features
with st.expander('Data preparation'):
    st.write('**Encoded input penguins**')
    st.dataframe(input_df)
    st.write('**Encoded X**')
    st.dataframe(X)
    st.write('**Encoded y**')
    st.dataframe(y)

# Load model
clf = get_model(X, y)

# Make prediction
predict_penguin(clf, input_df)
