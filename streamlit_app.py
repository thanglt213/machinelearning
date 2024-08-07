import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# File path for saving the trained model
MODEL_FILE_PATH = "penguin_random_forest_model.pkl"

st.title('🤖 Machine Learning App')

st.info(
    '''This is app builds a machine learning model in Python with streamlit!
    \n Source: 
    https://github.com/dataprofessor/dp-machinelearning/blob/master/streamlit_app.py
    https://www.youtube.com/@streamlitofficial
    '''
)

# Function to load data
@st.cache
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    X_raw = df.drop('species', axis=1)
    y_raw = df.species
    return df, X_raw, y_raw

# Get model function
@st.cache
def get_model():
    # Check if model file exists
    if os.path.exists(MODEL_FILE_PATH):
        # Load the model from file
        model = joblib.load(MODEL_FILE_PATH)
        st.success("Model loaded from file.")
    else:
        # Train the model and save it to file
        model = RandomForestClassifier()
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE_PATH)
        st.success("Model trained and saved to file.")
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
    # Add button to append data to the session state
    if st.button('Add data to predict'):
        # Combine with existing raw data (assuming X_raw is already defined)
        new_row = pd.DataFrame(data, index=[0])
        input_df = pd.concat([st.session_state.input_df, new_row], ignore_index=True)
        st.session_state.input_df = input_df

    return st.session_state.input_df

# Show raw data
with st.expander('Data'):
    st.write('**Raw data**')
    df, X_raw, y_raw = load_data()

    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    X_raw

    st.write('**y**')
    y_raw = df.species
    y_raw

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Initialize the session state to store the input data
if 'input_df' not in st.session_state:
    st.session_state['input_df'] = pd.DataFrame()
    input_df=st.session_state['input_df']
if 'input_penguins' not in st.session_state:
    st.session_state['input_penguins'] = pd.DataFrame()
    input_penguins=st.session_state['input_penguins']

with st.sidebar:
    st.header('Input features')
    get_user_input()
    
    """
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
    
    # Add button to append data to the session state
    if st.button('Add data to predict'):
        # Combine with existing raw data (assuming X_raw is already defined)
        new_row = pd.DataFrame(data, index=[0])
        input_df = pd.concat([st.session_state.input_df, new_row], ignore_index=True)
        st.session_state.input_df = input_df
    """

input_df = get_user_input()
input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
    st.write('**Input penguin**')
    input_df
    st.write('**Combined penguins data**')
    input_penguins

# Data preparation
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

n = len(input_df)
X = df_penguins[n:]
input_row = df_penguins[:n]

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y = y_raw.apply(lambda val: target_mapper[val])

with st.expander('Data preparation'):
    st.write('**Encoded X (input penguin)**')
    input_row
    st.write('**Encoded y**')
    y

def predict_penguin(clf, input_row: pd.DataFrame):
    if input_row.empty:
        st.write("Input data to predict!")
    else:
        prediction = clf.predict(input_row)
        prediction_proba = clf.predict_proba(input_row)
        
        # DataFrame containing prediction probabilities for each species
        df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])
        
        # Array containing the names of the penguin species
        penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
        
        # Add 'Predicted_Species' column to the DataFrame df_prediction_proba
        df_prediction_proba['Predicted_Species'] = [penguins_species[pred] for pred in prediction]
        
        # Display the DataFrame with prediction probabilities and predicted species
        st.subheader('Prediction Probabilities and Predicted Species')
        st.dataframe(df_prediction_proba, column_config={
            'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', width='medium', min_value=0, max_value=1),
            'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', width='medium', min_value=0, max_value=1),
            'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', width='medium', min_value=0, max_value=1),
            'Predicted_Species': st.column_config.TextColumn('Predicted Species', width='medium')
        }, hide_index=False)

# Load model
clf = get_model()
# Make prediction
predict_penguin(clf, input_row)


