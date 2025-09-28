import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    return df

data = load_data()

st.title("Dataset Insights & Random Forest Predictions")

# --- Show dataset ---
st.header("Raw Data")
st.dataframe(data)

# --- Show basic statistics ---
st.header("Data Summary")
st.write(data.describe())

# --- Show class distribution ---
st.header("Target Distribution")
sns.countplot(x='y', data=data)
st.pyplot(plt.gcf())
plt.clf()  # Clear figure for next plot

# --- Feature correlation heatmap ---
st.header("Feature Correlation")
corr = data.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot(plt.gcf())
plt.clf()

# --- Load trained model ---
@st.cache_resource
def load_model():
    return joblib.load("src/model/random_forest.pkl")

model = load_model()

# --- User input for prediction ---
st.header("Make a Prediction")

# Dynamically create input widgets for features
feature_inputs = {}
for col in data.drop(columns=['y']).columns:
    if data[col].dtype == 'object':
        options = data[col].unique().tolist()
        feature_inputs[col] = st.selectbox(f"Select {col}", options)
    else:
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        feature_inputs[col] = st.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=min_val)

# Button to predict
if st.button("Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([feature_inputs])

    # One-hot encode categorical columns like during training
    input_df = pd.get_dummies(input_df)
    # Add missing columns to match training data
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]  # Ensure correct column order

    prediction = model.predict(input_df)[0]
    st.success(f"The predicted class is: {prediction}")
