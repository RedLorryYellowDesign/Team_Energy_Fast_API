
# ---| ALL IMPORT LIBRARIES |---
# Base Streamlit Libraries
import requests # Allows use of URL imports
import streamlit as st # Allows compatibility with Streamlit
from streamlit_lottie import st_lottie # Allows lottie animation
from PIL import Image # Image manipulation
# Data Science Libraries
import plotly as py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Team_Energy.predict import forecast_model,evaluate,plot_graphs
from Team_Energy.data import create_data,get_weather
import joblib

# Page configuraion
st.set_page_config(page_title="Team Energy Le Wagon Project", page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")

# Use local css file to style the app
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style/style.css")

# fucntion to load lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ---| IMPORTING ASSEST |---
# Importing lottie animation
lottie_coding_Data_Science_Animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_xl3sktpc.json")


# ---| SEANORN DEFULT PLOT |---
x1 = np.linspace(0, 20, 100)
fig3 = sns.lineplot(data=x1)

# Header section
with st.container():
    Col_1,Col_2 = st.columns((3,1))
    with Col_1:
        st.title("Streamlit App")
        st.subheader("This is a Test web app for LeWagon Team Energy")
        st.write("This is a test app for LeWagon Team Energy to test Streamlit")
        st.write("[Le Wagon Home Page](https://www.lewagon.com)")

    with Col_2:
        st_lottie(lottie_coding_Data_Science_Animation, speed=1, height=200, key="initial")

# ---| USER INPUTS|---

with st.container():
    Col_1,Col_2 = st.columns((1,2))
    with Col_1:
        # tarrif
        st.subheader("Pease Select your Tarrif Type")
        User_Tarrif_Selected = st.selectbox('Pick one', ["","Fixed Tarrif", "Variable Tarrif"])

        if User_Tarrif_Selected == "Fixed Tarrif":
            User_Tarrif = 'Std'
        elif User_Tarrif_Selected == "Variable Tarrif":\
            User_Tarrif = 'ToU'
        else:
            st.write("Please select a tarrif")

        # Group
        st.subheader("Please Select your Group")
        User_Group_Selected = st.selectbox('Pick one', [" ","A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"])
        if User_Group_Selected != " ":
            st.write("Please select a Group")
        else:
            User_Group = User_Group_Selected

        # Submit Button
        if st.button("Submit"):
            st.write("Model will be called here")
            name=User_Group_Selected,tariff=User_Tarrif
            # Joblib import model
            filename = f'model_{name}_{tariff}.joblib'
            m = joblib.load(filename)
            #print('model loaded succcessfully')
            # ---| MODEL WILL BE CALLE HERE |---
            # def model_call(User_Tarrif_Selected,User_Group_Selected): # model will be called here
            #     return Model_Result
            train_df, test_df = create_data(name = name, tariff = User_Tarrif)
            train_wd, test_wd = get_weather(train_df, test_df)
            # Calculate forecast and MAPE
            forecast = forecast_model(m=m, train_wd = train_wd, test_wd = test_wd, add_weather = True)
            #print('forecast made')
            mape = evaluate(test_df['KWH/hh'], forecast['yhat'])
            confidence=(1-mape)*100
            # Plot the graphs
            #print('now plotting')
            plot_graphs(test_df = test_df, forecast= forecast)
            #print('operation complete')

            st.write('Your forecast accuracy is',confidence)
        else:
            st.write("Opps something went wrong")

# ---| DATA VISUALISATION |---

# plot sns line graph into streamlit
with Col_2:
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

# ---| FOOTER |---
# footer section with contrabuters and links
with st.container():
    st.write("---")
    col1, col2, col3= st.columns(3)
    with col1:
        st.write("The Team")
        st.write(" ")
    with col2:
        st.write(" ")
    with col3:
        st.write(" ")

# ---| END OF CODE |---
