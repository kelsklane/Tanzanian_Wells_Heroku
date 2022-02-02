import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
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
    st.markdown('This project was done to aid a fictional NGO in helping Tanzania accomplish its\' Millenium Development Goal of halving the proportion of the population without sustainable access to safe drinking water. At the time of the data being recorded, only 60% of Tanzanians could get their drinking water from an improved source. On top of that as of 2014, 33,200 rural water points were found to be non-functional with the likelihood of failure at 20% during these wells first year of operation. Therefore, access to working wells that deliver clean water is of high importance to Tanzania.')
    st.image('Images/6in10.png')
    st.markdown('46% of the wells in Tanzania are in need of repair or nonfunctioning. Instead of building new wells, this NGO can drastically increase clean water supply by fixing these broken wells. Given an NGO\'s financial constraints, sending a repair team out to a fully-functional well would be expensive and cost the opportunity of fixing an actual non-functioning well. This problem may be exacerbated since some of these wells are very remote and in mountainous regions. By using our model that reached a precision of 76%, the NGO can preemptively address probable pump failures, increasing the sustainable, improved well capacity of Tanzania.')
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
    st.markdown('To get a sense of the data, most of it was collected between 2011 - 2013, with one record dating from 2002 and 30 from 2004. The map below shows the spread of wells across Tanzania, where the blank spots with no wells are in sparsely populated areas in the desert.')
    map_data = df_trim[['longitude', 'latitude']]
    tanz_map = folium.Map(location=[map_data.latitude.mean(), 
                           map_data.longitude.mean()], zoom_start=6, control_scale=True,
                     tiles = "Stamen Terrain")
    heat_data = [[row['latitude'],row['longitude']] for index, row in map_data.iterrows()]
    HeatMap(heat_data).add_to(tanz_map)
    folium_static(tanz_map)

    #Histogram of tsh_amount
    st.markdown('One of the variables we found to be significant in predicting well status is `amount_tsh` or the total static head in the well. This measure represents the distance the water would have to travel to be pumped out of the well and thus is a metric used to gauge a pump\'s ability to pump water. The distribution of this variable in the given data is displayed below.')
    atsh = df_trim[df_trim.amount_tsh < 30]
    fig6, ax6 = plt.subplots()
    sns.set_style("ticks")
    sns.histplot(data = atsh, x = 'amount_tsh', kde = True, ax = ax6)
    ax6.set_title('Distribution of Total Static Head Amounts')
    ax6.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax6.set_ylabel('Number of Wells')
    ax6.set_xlabel('Amount of Total Static Head')
    st.pyplot(fig6)
    
    #Bar chart for permit/installer/extraction type
    #Permit
    st.markdown('Another significant variable in predicting a well\'s status is if it had recieved a permit from the government. Some of the wells were built without government approval and these tended to have a higher rate of failure. The distribution of this variable in the training data is graphed below as well.')
    fig2, ax2 = plt.subplots()
    sns.set_style("ticks")
    sns.barplot(x = ['No Permit', 'Has Permit'], y = np.bincount(df_trim.permit), color = '#FFAC1C', ax = ax2)
    ax2.set_title('Distribution of Government Permits For Wells')
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax2.set_ylabel('Number of Permits')
    ax2.set_xlabel('Permit Status')
    ax2.bar_label(ax2.containers[0])
    st.pyplot(fig2)
    #Installer
    st.markdown('The group that installed the well is another significant factor in the model for predicting the well\'s status. These installers include the District Water Engineers (DWE), Regional Water Engineer\'s Office (RWE), Denmark\'s Development Cooperation (DANIDA), Commu, government-sponsored wells, as well as a collection of wells installed by either smaller groups or of unknown origin lumped under other.')
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
    st.markdown('The final significant variable is the method the well uses to extract the water from the ground. These include self-explanatory methods like gravity-operated wells, handpumps that require manual pumping, motorpumps, and rope-operated wells. Other extraction methods include wind-powered, where a wind turbine is used to generate the energy needed to pull up the water, and submersible pumps, where a submerged pumping unit is used to draw up water. Finally, any unknown or uncommon well extraction types are lumped together under the other category.')
    ext = df_trim.extraction_type_class.value_counts(ascending = False)
    index = ext.index
    values = ext.values
    fig4, ax4 = plt.subplots(figsize = (12, 8))
    sns.set_style("ticks")
    sns.barplot(x = index, y = values, color = '#FFAC1C', ax = ax4)
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
    tsh = form.slider("Choose a total static head amount: ", min_value=0.0, max_value = 30.0, value=15.0, step=.01)
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
