import streamlit as st
import pandas as pd

st.title('🎈 Machine learning app')

st.info('This is a machine learning app from https://www.youtube.com/@streamlitofficial')

with st.expander('Data'):
  st.write('**Raw data**')
  df=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

  st.write("**X**")
  X=df.drop('species', axis=1)
  X
  st.write("**y**")
  y=df['species']
  y
with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Data preparation
island_list=['Torgersen', 'Biscoe', 'Dream']
with st.sidebar:
  st.write('Input features:')
  island = st.selectbox('Island:',island_list)
  gender = st.selectbox('Gender:',['male','female'])
  bill_length_mm = st.slider('Bill length in mm:',32.1,59.6,44.0)
  bill_depth_mm = st.slider('Bill depth in mm:',13.1,21.5,17.1)
  flipper_length_mm = st.slider('Flipper length in mm:', 172,231,200)
  body_mass_g = st.slider('Body mass in g:',2700,6300,4207)
