import pandas as pd
import numpy as np
import pickle
import string
import streamlit as st
from PIL import Image

  
#Load in model
pickle_in = open('classifier.pkl', 'rb')
well_class = pickle.load(pickle_in)

#Function to make prediction
@st.cache
def prediction(t, p, i, e):  
    df = pd.DataFrame([[t, p, i, e]], columns=['amount_tsh', 'permit', 'installer', 'extraction_type_class'])
    prediction = well_class.predict(df)
    prob = well_class.predict_proba(df)
    return prediction[0], prob

page = st.sidebar.selectbox("Choose a page:", ["Data Visualization", "Well Prediction"]) 

if page == 'Data Visualization':
    st.title("Tanzanian Well Data Visualizations")
    #Load the data

    #Functionality bar chart
    #6 in 10 get drinking water from clean source
    #Map of wells w/ altitude as darkness of dot
    #Histogram of tsh_amount
    #Bar chart for permit/installer/extraction type
    #Change in functionality over time



elif page == 'Well Prediction':
    st.title("Well Status Prediction")
    st.subheader('Test different well parameter combinations to predict if the well would be functioning or not!')
    st.image('Images/unnamed.jpg')
    form = st.form(key = "text_form")
    tsh = form.slider("Choose a total static head amount: ", min_value=0.0, max_value=30.0, value=15.0, step=.01)
    permit = form.radio('Does the well have a permit?', options = ['True', 'False'])
    installer = form.selectbox('Who was the installer for the well?', options = ['other', 'dwe', 'government', 'rwe', 'danida', 'commu'])
    extract = form.selectbox('What was the extraction type used for the well?', options = ['gravity', 'handpump', 'other', 'submersible', 'motorpump', 'rope pump', 'wind-powered'])
    submit = form.form_submit_button(label="Predict Well Status")


    if submit:
        if permit == 'True':
            permit = True
        else:
            permit = False

        #Make prediction from the input text
        result, prob = prediction(t = tsh, p = permit, i = installer, e = extract)

        #Print prediction
        st.write("**Result:**")
        if result == 0:
            pb = prob[0][0] * 100
            pb = round(pb, 2)
            st.write(f'There is a {pb}% chance that this well is functional!')
        else:
            pb = prob[0][1] * 100
            pb = round(pb, 2)
            st.write(f'There is a {pb}% chance that this well needs repair, better send somebody out.')
