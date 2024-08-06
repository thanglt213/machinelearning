
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# File path for saving the trained model
MODEL_FILE_PATH = "penguin_random_forest_model.pkl"
DATA_FILE_PATH = "penguins_data.csv"

# Title
st.title('🤖 Machine Learning App')

st.info(
    '''This app builds a machine learning model in Python with Streamlit!
    \n Source: 
    https://github.com/dataprofessor/dp-machinelearning/blob/master/streamlit_app.py
    https://www.youtube.com/@streamlitofficial
    '''
)

# Check if data is available in session state, otherwise load it
if 'df' not in st.session_state:
    if os.path.exists(DATA_FILE_PATH):
        df = pd.read_csv(DATA_FILE_PATH)
        st.success("Data loaded from file.")
    else:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
        df.to_csv(DATA_FILE_PATH, index=False)
        st.success("Data loaded from source and saved to file.")
    st.session_state['df'] = df
else:
    df = st.session_state['df']

# Display raw data
with st.expander('Data'):
    st.write('**Raw data**')
    st.write(df)

    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    st.write(X_raw)

    st.write('**y**')
    y_raw = df.species
    st.write(y_raw)

# Data visualization
with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Sidebar for input features
with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))

    input_data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame(input_data, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

    # Add button to add new data to the dataset
    if st.button('Add data'):
        new_row = pd.DataFrame(input_data, index=[len(df)])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_FILE_PATH, index=False)
        st.session_state['df'] = df
        st.success('New data added to the dataset.')

# Data preparation
encode = ['island', 'sex']
df_penguins = pd.get_dummies(df, columns=encode)
input_penguins_encoded = pd.get_dummies(input_penguins, columns=encode).reindex(columns=df_penguins.columns, fill_value=0)

X = df_penguins.drop(columns='species')
y = y_raw.apply(lambda val: {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}[val])

with st.expander('Data preparation'):
    st.write('**Encoded X (input penguin)**')
    st.write(input_penguins_encoded.head(1))
    st.write('**Encoded y**')
    st.write(y)

# Check if model file exists
if os.path.exists(MODEL_FILE_PATH):
    # Load the model from file
    clf = joblib.load(MODEL_FILE_PATH)
    st.success("Model loaded from file.")
else:
    # Train the model and save it to file
    clf = RandomForestClassifier()
    clf.fit(X, y)
    joblib.dump(clf, MODEL_FILE_PATH)
    st.success("Model trained and saved to file.")

# Apply model to make predictions on the input data
predictions = clf.predict(input_penguins_encoded)
prediction_probas = clf.predict_proba(input_penguins_encoded)

# Display predicted species
df_prediction_proba = pd.DataFrame(prediction_probas, columns=['Adelie', 'Chinstrap', 'Gentoo'])
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba, hide_index=True)

predicted_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])[predictions]
st.success(f"Predicted species for input data: {predicted_species[0]}")
