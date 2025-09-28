import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Subscription Prediction Dashboard",
    page_icon=":bank:",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("src/model/random_forest.pkl")
        return model
    except FileNotFoundError:
        return None
    
@st.cache_resource
def load_training_columns():
    try:
        columns = joblib.load("src/model/training_columns.pkl")
        return columns
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/train.csv")
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_eval_data():
    try:
        y_val = joblib.load("src/model/y_val.pkl")
        y_pred = joblib.load("src/model/y_pred.pkl")
        return y_val, y_pred
    except FileNotFoundError:
        return None 

