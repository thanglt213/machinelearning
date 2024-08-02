import streamlit as st
import pandas as pd

st.title('🎈 Machine learning app')

st.info('This is a machine learning app')

# df=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
df=pd.read_csv("https://github.com/dataprofessor/data/blob/master/penguins_cleaned.csv")
df
