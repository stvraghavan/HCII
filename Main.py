from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

dataframe = pd.DataFrame()

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

col1,col,col3 = st.tabs(['First Look','Descriptive Statistics','Charts'])

with col1:
    st.write("A snap of the dataframe")
    st.write(dataframe)
with col2:
    st.write(dataframe.describe())