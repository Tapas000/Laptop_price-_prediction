import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Ensure df is a DataFrame
if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)

st.title("💻 Laptop Price Prediction")

# User inputs
company = st.selectbox('Brand', df['Company'].unique())
Type = st.selectbox("Laptop Type", df['TypeName'].unique())
RAM = st.selectbox("RAM (GB)", df['Ram'].unique())
memory = st.selectbox("Storage (GB)", df['Memory'].unique())
weight = st.number_input("Weight (kg)")
Touch_screen = st.radio("Touch Screen", ['Yes', 'No'])
full_hd = st.radio("Full HD Display", ['Yes', 'No'])
Ips = st.radio("IPS Panel", ['Yes', 'No'])
screen_size = st.number_input("Screen Size (inches)")
resolution = st.selectbox("Screen Resolution", [
    "1366x768", "1920x1080", "2560x1440", "2560x1600",
    "2880x1800", "3200x1800", "3840x2160"
])
Cpu_brand = st.selectbox("CPU Brand", df['Cpu Brand'].unique())
Gpu_brand = st.selectbox("GPU Brand", df['Gpu brand'].unique())
Os = st.selectbox("Operating System", df['Operating system'].unique())

if st.button("🔍 Predict Price"):
    # Convert categorical Yes/No values to binary
    Touch_screen = 1 if Touch_screen == 'Yes' else 0
    full_hd = 1 if full_hd == 'Yes' else 0
    Ips = 1 if Ips == 'Yes' else 0

    # Extract screen resolution
    x_res, y_res = map(int, resolution.split('x'))
    ppi = ((x_res**2 + y_res**2) ** 0.5) / screen_size

    # Prepare input array
    query = np.array([company, Type, RAM, memory, weight, Touch_screen,
                      full_hd, Ips, ppi, Cpu_brand, Gpu_brand, Os]).reshape(1, -1)

    # Convert to DataFrame with proper column names
    column_names = ['Company', 'TypeName', 'Ram', 'Memory', 'Weight', 'Touchscreen',
                    'full_HD', 'IPS', 'ppi', 'Cpu Brand', 'Gpu brand', 'Operating system']


    query_df = pd.DataFrame(query, columns=column_names)

    # Ensure all column names are strings
    query_df.columns = query_df.columns.astype(str)

    # Predict price
    predicted_price = np.exp(pipe.predict(query_df)[0])  # Remove np.exp() if not log-transformed

    # Display prediction
    st.success(f"💲 The estimated price of this laptop is **₹{int(predicted_price)}**")
