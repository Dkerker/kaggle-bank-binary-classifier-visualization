# Copyright 2025 Jacob Groll
#
# app.py
#
# This script creates a Streamlit dashboard to visualize the bank marketing dataset
# and display the performance of a trained machine learning model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

model = load_model()
training_columns = load_training_columns()
df_train = load_data()
y_val, y_pred = load_eval_data()

if model is None or df_train is None or y_val is None:
    st.error(
        "**Error:** Model or data files not found. "
    )
    st.stop()

label_map = {1: 'Subscribed', 0: 'Not Subscribed', 'yes': 'Subscribed', 'no': 'Not Subscribed'}
df_train['subscription_status'] = df_train['y'].map(label_map)

color_map = {'Not Subscribed': '#68a0d0', 'Subscribed': '#00429d'}

st.title(":bank: Bank Customer Prediction Dashboard")
st.markdown("""
    Analyzes customer data and evaluates the performance of a model
    trained to predict customer subscriptions.
""")

tab1, tab2, tab3 = st.tabs([":bar_chart: Data Overview", ":brain: Model Performance", ":clipboard: Raw Data Snapshot"])

with tab1:
    st.header("Customer Data Overview")

    col1, col2, = st.columns(2)
    with col1:
        st.subheader("Subscription Status")
        status_counts = df_train['subscription_status'].value_counts()
        fig_pie = px.pie(
            names=status_counts.index,
            values=status_counts.values,
            title="Distribution of Subscriptions",
            color=status_counts.index,
            color_discrete_map=color_map
        )
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Job Distribution")
        job_counts = df_train['job'].value_counts().nlargest(10)
        fig_bar_job = px.bar(
            x=job_counts.index,
            y=job_counts.values,
            title="Top 10 Customer Professsions",
            labels={'x': 'Job', 'y': 'Count'}
        )
        st.plotly_chart(fig_bar_job, use_container_width=True)

    st.subheader("Customer Age Distribution")
    fig_hist_age = px.histogram(
        df_train,
        x='age',
        nbins=50,
        title="Distribution of Customer Ages",
        color='subscription_status',
        color_discrete_map=color_map
    )
    st.plotly_chart(fig_hist_age, use_container_width=True)


with tab2:
    st.header("Model Perfomance Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Metrics")
        accuracy = accuracy_score(y_val, y_pred)
        st.metric(label="Validation Accuracy", value=f"{accuracy:.2%}")

        st.subheader("Classification Report")
        y_val_mapped = pd.Series(y_val).map(label_map)
        y_pred_mapped = pd.Series(y_pred).map(label_map)
        report = classification_report(y_val_mapped, y_pred_mapped, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(2))

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_val, y_pred)
        labels = [label_map[label] for label in model.classes_]

        fig_cm = ff.create_annotated_heatmap(
            z=cm, x=labels, y=labels, colorscale='Blues', showscale=True
        )
        fig_cm.update_layout(
            title_text = '<b>Confusion Matrix<b>',
            xaxis = dict(title='Predicted Value'),
            yaxis = dict(title='Actual Value')
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.header("Model Feature Importance")
    st.markdown("This chart shows the most influential factors the model uses to make predictions")

    if training_columns is not None:
        feature_importance = pd.DataFrame({
            'feature': training_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        fig_importance = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 15 Most Important Features",
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)

with tab3:
    st.header("Full Training Dataset")
    st.dataframe(df_train)

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.9em; color: #808080;">
    <p>This dashboard was created using the <b>Bank Marketing Dataset</b> from a Kaggle competition.</p>
    <p>The dataset is available under the <a href="http://www.apache.org/licenses/LICENSE-2.0" target="_blank">Apache 2.0 License</a>.</p>
</div>
""", unsafe_allow_html=True)