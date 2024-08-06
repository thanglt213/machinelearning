import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

st.title('🤖 Machine Learning App')

st.info(
    '''This is app builds a machine learning model in Python with streamlit!
    \n Source: 
    https://github.com/dataprofessor/dp-machinelearning/blob/master/streamlit_app.py
    https://www.youtube.com/@streamlitofficial
    '''
)

# File path for saving the trained model
MODEL_FILE_PATH = "penguin_random_forest_model.pkl"

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    df

    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    X_raw

    st.write('**y**')
    y_raw = df.species
    y_raw

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')


# Initialize the dataframe if not present in session state
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])

data = st.session_state['data']

# Sidebar for input features
with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))

    new_row = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }

    if st.button('Add data'):
        # Check if the new_row has the same columns as the existing data
        try:
            st.session_state['data'] = pd.concat([st.session_state['data'], pd.DataFrame([new_row])], ignore_index=True)
            st.success('New data added.')
        except Exception as e:
            st.error(f"Error adding data: {e}")

# Display current data
st.write("Current data in the session:")
st.dataframe(st.session_state['data'])

# Prepare data for modeling
if not st.session_state['data'].empty:
    try:
        encode = ['island', 'sex']
        df_encoded = pd.get_dummies(st.session_state['data'], columns=encode)

        # Ensure the necessary columns are available for prediction
        if os.path.exists(MODEL_FILE_PATH):
            clf = joblib.load(MODEL_FILE_PATH)
            st.success("Model loaded from file.")
        else:
            clf = RandomForestClassifier()
            # Dummy training for demonstration purposes (replace with actual training)
            X_train, y_train = df_encoded, [0] * len(df_encoded)  # Placeholder
            clf.fit(X_train, y_train)
            joblib.dump(clf, MODEL_FILE_PATH)
            st.success("Dummy model trained and saved to file.")

        # Apply model to make predictions on the data
        predictions = clf.predict(df_encoded)
        st.write("Predictions:")
        st.write(predictions)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.warning("No data available for predictions.")
