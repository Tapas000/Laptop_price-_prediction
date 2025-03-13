import columns
from lib2to3.fixer_util import touch_import

import streamlit as st
import pickle
import numpy as np
import  pandas as pd

pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)


st.title("Laptop Price prediction")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop

Type = st.selectbox("type",df['TypeName'].unique())
# ram
RAM = st.selectbox("Ram",df['Ram'].unique())

memory = st.selectbox("Memory",df['Memory'].unique())

# weight
weight = st.number_input("Weight")

# touch screen
Touch_screen = st.selectbox("Touch Screen",['Yes','No'])

full_hd = st.selectbox("Full HD",['Yes','No'])
# ips
Ips = st.selectbox("IPS",['Yes','No'])

screen_size = st.number_input("screen size")

resolution = st.selectbox("Screen resolution",["1366x768",
    "1920x1080",
    "2560x1440",
    "2560x1600",
    "2880x1800",
    "3200x1800",
    "3840x2160"])

#cpu brand
Cpu_brand = st.selectbox("Cpu_brand",df['Cpu Brand'].unique())

Gpu_brand = st.selectbox("Gpu_brand",df['Gpu brand'].unique())

Os = st.selectbox("Operating System",df['Operating system'].unique())

if st.button("Predict Price"):
    if Touch_screen == 'Yes':
        Touch_screen =1
    else:
        Touch_screen = 0

    if Ips == 'Yes':
        Ips = 1
    else:
        Ips = 0

    if full_hd == 'Yes':
        full_hd = 1
    else:
        full_hd =0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2)+(y_res**2))**0.5/screen_size

    query = np.array([company,Type,RAM,memory,weight,Touch_screen,full_hd,Ips,ppi,Cpu_brand,Gpu_brand,Os])
    query= query.reshape(1,12)


    st.title(np.exp(pipe.predict(query)))