# Load packages
import pandas as pd
import numpy as np
import nbformat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.subplots as sp
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import datetime
import streamlit as st
import joblib

# ml packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import make_scorer
from math import sqrt

# remove warnings
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    layout="wide"
)

# -------------------------------------------------------------
# Intro & Objective
# -------------------------------------------------------------
st.title("🏠 Predicting HDB Resale Flat Prices in Singapore")
st.write('')
st.write('')

col1, col2 = st.columns([1,1])  

with col1:
    st.image(
        'images/hdb-flat.jpg', 
        caption="HDB Flat in Yishun, Singapore (📸: me)",
        width=500  
    )

with col2:
    st.write("""
    **Objective**: To predict Singapore HDB resale flat prices using machine learning techniques.
    """)

    st.write("""
    _Note:_  
    - _Random forest was chosen for its interpretability and performance across RMSE, MAE, R²_ 
    - _Predictions focus on 2024 resale prices, based on 2023 and 2024 resale transactions_  
    - _Input ranges are restricted to values observed in the training data for consistency_
    - _Analysis and insights are available in the `Analysis & Insights` page via left tab_
    """)

st.write('---')


# -------------------------------------------------------------
# Model #1 - Inference
# -------------------------------------------------------------
rf_model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoded_columns = joblib.load('models/encoded_columns.pkl')

df = pd.read_csv('data-github/geocoded_data_with_hdb_property_info_processed.csv')

numerical_predictors = ['remaining_total_years', 'floor_area_sqm']
categorical_predictors = ['Region', 'storey_range']

st.write("## 📍 Predict HDB Resale Price")
# st.write("#### Enter resale flat details:")
st.write("""
_Typical floor area (sqm): 1 Room: 30-50, 2 Room: 45-70, 3 Room: 60-90, 4 Room: 85-110, 5 Room: 110-130, Executive: 130+_
""")

user_input = {}
cols = st.columns(len(numerical_predictors)) 
for idx, num_feature in enumerate(numerical_predictors):
    with cols[idx]:  
        min_value = df[num_feature].min()
        if num_feature == 'floor_area_sqm':
            max_value = 150
        else:
            max_value = 100
        default_value = max_value / 2
        user_input[num_feature] = st.slider(
            f"{num_feature.replace('_', ' ').capitalize()}:",
            min_value=int(1),
            max_value=int(max_value),
            value=int(default_value)
        )

for cat_feature in categorical_predictors:
    unique_values = df[cat_feature].dropna().unique().tolist()
    default_value = unique_values[0]
    user_input[cat_feature] = st.segmented_control(
        label=f"{cat_feature.replace('_', ' ').capitalize()}",
        options=unique_values,
        default = default_value
    )

user_df = pd.DataFrame([user_input])
user_df_encoded = pd.get_dummies(user_df, columns=categorical_predictors, drop_first=True)
user_df_encoded = user_df_encoded.reindex(columns=encoded_columns, fill_value=0)
user_df_encoded[numerical_predictors] = scaler.transform(user_df_encoded[numerical_predictors])
st.write('')
st.write('')


st.markdown("""
    <div style="text-align: center;">
        <p><strong>Enter details above and click the button to predict resale price!</strong></p>
    </div>
""", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3:
    center_button = st.button("Predict Resale Price 🎈")

    if center_button:
        predicted_price = rf_model.predict(user_df_encoded) 
        st.write(f"### SGD ${predicted_price[0]:,.0f}")
        st.balloons()

st.write("---")


# -------------------------------------------------------------
# Overview & Details
# -------------------------------------------------------------
st.write('## Overview: Random Forest Regression')
st.write("""
This model focuses on predicting HDB resale flat prices using a random forest regression approach. 
It leverages data from the latest 2 years (2023/24), ignoring the time-dependent complexities such as seasonality or long-term trends. 
The aim is to provide an initial prediction framework before incorporating more advanced time-series techniques and accounting for temporal dependencies.
- **Target Variable**: `resale_price`
- **Predictors**:
  - Numerical: `remaining_total_years`, `floor_area_sqm`
  - Categorical: `region`, `storey_range`
- **Machine Learning Algorithms Tested**: Linear Regression, Decision Trees, Random Forest Regression, XGBoost
- **Performance Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R²
""")
st.write('')
st.write("#### Model evaluation")
st.write('')
# -------------------------------------------------------------
#    Model evaluation & feature importance
# -------------------------------------------------------------

col1, gap, col2 = st.columns([1,0.1,1])  

with col1:
    results_df = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
        "RMSE": [107522, 92361, 79242, 76512],  
        "MAE": [84878, 64737, 57091, 58036],  
        "R²": [0.63, 0.73, 0.80, 0.81]        
    })
    results_df["RMSE"] = results_df["RMSE"].apply(lambda x: f"{x:,}")
    results_df["MAE"] = results_df["MAE"].apply(lambda x: f"{x:,}")
    results_df["R²"] = results_df["R²"].apply(lambda x: f"{x:.2f}")
    st.write("##### Scoring the models")

    st.write('Best performing model with interpretability: **Random Forest**')
    st.table(results_df.style.set_table_attributes('style="width:50%"'))  

# -------------------------------------------------------------
#    Feature importance
# -------------------------------------------------------------
with col2:
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(encoded_columns)[sorted_idx]
    sorted_importances = importances[sorted_idx]

    rf_importances_df = pd.DataFrame({
        'Feature': sorted_features,
        'Importance': sorted_importances
    })

    st.write("##### Feature importance")
    st.write('These are the top 20 features with the most predictive power in the Random Forest model. `Floor_area_sqm` contributes the most (~58%), followed by `remaining_total_years` (~20%)')

    # top_n = st.slider("Select number of features to display:", min_value=5, max_value=20, value=10)

    fig = px.bar(
        rf_importances_df.head(20),
        x='Importance',
        y='Feature',
        orientation='h',
        # title="",
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        width=600,
        height=600
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor="rgba(200, 200, 200, 0.3)",  
            griddash="dot" 
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show feature importance table"):
        st.write(rf_importances_df.head(20))

# -------------------------------------------------------------
#    Data Preview
# -------------------------------------------------------------
st.write('#### A look at our dataset')
st.write('Download the processed data @ `data/geocoded_data_with_hdb_property_info_processed.csv` ')
st.write(df.head(5))

# -------------------------------------------------------------
#   Data Processing
# -------------------------------------------------------------
st.write('#### Preparing the data')
st.write("""
For more details on data processing, refer to `workings.ipynb` file
- data type standardization
- joining multiple data sources
- geocoding
- handle missing data and outliers
- feature engineering
- data transformation
- binning
- scaling numerical features
- encoding categorical features
- addressing multicollinearity
- applying domain knowledge
""")

# -------------------------------------------------------------
#    Data Sources
# -------------------------------------------------------------
st.write('#### References')
st.write("""
- HDB resale data: https://data.gov.sg/collections/189/view
- HDB property information data: https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view
- Geocoding API: https://www.onemap.gov.sg/apidocs/search
- Procedures to buy a resale flat: https://www.hdb.gov.sg/business/estate-agents-and-salespersons/buying-a-resale-flat
""")


# -------------------------------------------------------------
# Contact
# -------------------------------------------------------------
st.write('')
st.write('')
st.markdown("""
    <style>
        .footer {
            bottom: 10px;
            left: 0;
            right: 0;
            text-align: center;
            font-size: 12px;
            color: gray;
            margin-top: 20px;
        }
    </style>
    <div class="footer">
        Author: Darryl Lee | 
        <a href="https://www.linkedin.com/in/darryl-lee-jk/" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/darrylljk" target="_blank">GitHub</a>
    </div>
""", unsafe_allow_html=True)