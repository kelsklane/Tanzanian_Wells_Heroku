import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import plotly.figure_factory as ff
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
    df_feat = pd.read_csv('data/Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_values.csv')
    df_targ = pd.read_csv('data/Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv')
    df = pd.concat([df_feat, df_targ], axis = 1)
    df_trim = df.drop(columns = ['id', 'extraction_type', 'extraction_type_group', 'scheme_name',
                            'payment', 'quality_group', 'quantity_group', 'source_type', 'waterpoint_type_group',
                            'region_code', 'district_code', 'ward', 'subvillage', 'lga', 'num_private',
                            'recorded_by', 'funder', 'public_meeting', 'wpt_name'], axis = 1)
    df_trim = df_trim[df_trim['longitude'] != 0]
    df_trim['permit'] = df_trim['permit'].fillna(value = False)
    def decades(year):
        if year == 0:
            return 'Unknown'
        else:
            return str((year // 10) * 10)
    df_trim['construction_year'] = df_trim['construction_year'].apply(decades)

    df_trim['month'] = pd.DatetimeIndex(df_trim['date_recorded']).month
    season = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 
            8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
    df_trim['season'] = df_trim['month'].map(season)
    df_trim = df_trim.drop(columns = ['month', 'date_recorded'], axis = 1)
    df_trim.status_group = df_trim.status_group.map({'functional' : 0, 'non functional': 1, 'functional needs repair': 1})
    #Bin installer column on training frequency
    def install_bin(entry):
        #Bins nulls as other
        if type(entry) == float:
            return 'other'
        #Checks lowercase to account for mistyped entries
        elif entry.lower() in inst_list:
            return entry.lower()
        else:
            return 'other'

    #Bin scheme_management based on training frequency
    def scheme_bin(entry):
        if type(entry) == float:
            return 'other'
        elif entry.lower() in scheme_list:
            return entry.lower()
        else:
            return 'other'

    #Bin and fill in nulls in installer
    inst_five = df_trim.installer.value_counts(sort = True, ascending = False)[:5]
    inst_list = list(inst_five.index)
    for idx, value in enumerate(inst_list):
        inst_list[idx] = value.lower()
    df_trim['installer'] = df_trim['installer'].apply(install_bin)
        
    #Bin and fill in nulls in scheme_management
    scheme_eight = df_trim.scheme_management.value_counts(sort = True, ascending = False)[:9]
    scheme_list = list(scheme_eight.index)
    for idx, value in enumerate(scheme_list):
        scheme_list[idx] = value.lower() 
    df_trim['scheme_management'] = df_trim['scheme_management'].apply(scheme_bin)

    #Functionality bar chart
    st.markdown('Background about the project.')
    st.image('Images/6in10.png')
    st.markdown('Get a sense of what the data worked with like see target distribution below.')
    fig, ax = plt.subplots()
    sns.set_style("ticks")
    sns.barplot(x = ['Functional', 'Non-functional'], y = np.bincount(df_trim.status_group), color = '#0072b2')
    ax.set_title('Number of Functional and Non-functional Wells in the Given Data')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_ylabel('Number of Wells')
    ax.set_xlabel('Well Status')
    ax.bar_label(ax.containers[0])
    st.pyplot(fig)
    #Map of wells w/ altitude as darkness of dot

    #MENTION TIME - MOST DATA COLLECTED 2011-2013 w/ 1 from 2002 and 30 from 2004

    #Histogram of tsh_amount

    # st.markdown('Describe what amount_tsh is and how it would impact well performance')
    # tsh_data = list(df_trim.amount_tsh)
    # hist_data = [tsh_data]
    # group_labels = ['amount_tsh']
    # fig2 = ff.create_distplot(hist_data, group_labels, bin_size = 10)
    # st.plotly_chart(fig2, use_container_width=True)
    
    #Bar chart for permit/installer/extraction type
    #Permit
    st.markdown('Background about the project. Below is a graph of the initial distribution of the target in the data used to train the model to get a sense of the status of Tanzanian wells.')
    fig2, ax2 = plt.subplots()
    sns.set_style("ticks")
    sns.barplot(x = ['No Permit', 'Has Permit'], y = np.bincount(df_trim.permit), color = '#f07167', ax = ax2)
    ax2.set_title('Distribution of Government Permits For Wells')
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax2.set_ylabel('Number of Permits')
    ax2.set_xlabel('Permit Status')
    ax2.bar_label(ax2.containers[0])
    st.pyplot(fig2)
    #Installer
    st.markdown('Talk about installers')
    inst = df_trim.installer.value_counts(ascending = False)
    index = inst.index
    values = inst.values
    fig3, ax3 = plt.subplots()
    sns.set_style("ticks")
    sns.barplot(x = index, y = values, color = '#0072b2', ax = ax3)
    ax3.set_title('Distribution of Well Installers')
    ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax3.set_ylabel('Quantity of Wells Installed')
    ax3.set_xlabel('Installer')
    ax3.bar_label(ax3.containers[0])
    st.pyplot(fig3)
    #Extraction
    st.markdown('Talk about well extractions')
    ext = df_trim.extraction_type_class.value_counts(ascending = False)
    index = ext.index
    values = ext.values
    fig4, ax4 = plt.subplots(figsize = (12, 8))
    sns.set_style("ticks")
    sns.barplot(x = index, y = values, color = '#f07167', ax = ax4)
    ax4.set_title('Distribution of Well Extraction Types')
    ax4.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax4.set_ylabel('Quantity of Each Extraction')
    ax4.set_xlabel('Extraction Type')
    ax4.bar_label(ax4.containers[0])
    st.pyplot(fig4)



elif page == 'Well Prediction':
    st.image('Images/well_2.jpeg')
    st.title("Well Status Prediction")
    st.subheader('Test different well parameter combinations to predict if the well would be functioning or not!')
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
