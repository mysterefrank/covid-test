import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
import streamlit as st
from PIL import Image
from model_funcs import *
import datetime
import covid.util as util
from os import path


currently_infected = 511

st.sidebar.title("Neurite Interface v0.1")

st.sidebar.subheader("Training Parameters")

start_date = st.sidebar.date_input('Start Date', datetime.date(2020,3,4))
end_date = st.sidebar.date_input('End Date', datetime.date(2020,8,2))
location = st.sidebar.selectbox('City to forecast',
                                ('NY', 'NC', 'FL', 'CA'))

train_event = st.sidebar.button('Train model')

st.sidebar.subheader("Forecasting Parameters")

forecast_horizon_days = st.sidebar.number_input(
    "How many days out to forecast", value=25, step=5, format="%i"
)

forecast_event = st.sidebar.button('Generate Forecasts')

if train_event:
    with st.spinner('Training...'):
        train_model(start_date, end_date, location)
    st.balloons()

if forecast_event:
    with st.spinner('Generating Forecasts...'):
        generate_forecasts(start_date, location, forecast_horizon=forecast_horizon_days)
    st.balloons()
    image = Image.open(f'./results/vis/{location}_scale_lin_daily_True_T_{forecast_horizon_days}.png')

st.title("Forecasts")


st.markdown(
f"""The estimated number of currently infected individuals is {currently_infected}, this is a 5% reduction over 
the previous week. R0 is 0.8. 
"""
)

# Load plot
initial_plot = f'./results/vis/{location}_scale_lin_daily_True_T_{forecast_horizon_days}.png'
if path.exists(initial_plot):
    image = Image.open(initial_plot)
    st.image(image)

