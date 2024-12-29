# HDB Resale Price Predictor

I've created this app to help make predicting Singapore's HDB resale flat prices easier and more accessible. By leveraging Random Forest regression, the prediction model provides reliable price estimates based on the flat details you input. 

Beyond predictions, the app includes a variety of visualizations and analyses to shed light on the factors influencing resale prices. Whether you're exploring the market as a buyer, seller, or just curious, this tool is designed to offer useful insights into Singapore's housing landscape.

## Try it out
- Access the web app here: [HDB Resale Price Predictor](https://hdb-resale-predictor.streamlit.app/) 🚀
- To run locally:
  - clone the repository: `git clone https://github.com/darrylljk/HDB-Resale-Price-Predictor.git`
  - install dependencies: `pip install -r requirements.txt`
  - launch streamlit app: `streamlit run HDB_Resale_Price_Predictor.py`

## App Modules
### 1. HDB Resale Price Predictor (Random Forest)
Accurately predict Singapore HDB resale flat prices using Random Forest regression model
- Predictive model trained on 2023 and 2024 HDB resale transaction data
- Incorporates features such as floor area, region, and storey range
- Simple and intuitive interface for inputting flat details and generating predictions
- Random Forest outperforms other models (linear regression, decision tree, XGboost)
- Model evaluation (scoring, feature importance)

### 2. Analysis & Insights
Dive into the key factors influencing HDB resale prices through interactive and insightful visualizations.
- resale flat prices across SG
- distribution of flat types, models, resale prices
- resale prices by location (town and region)
- resale price trends and YoY % change
- impact of features (floor area, storey range, remaining lease) on resale price
- impact of location on price per sqm
- correlation between features
and more

## Gallery
### 1. HDB Resale Price Predictor (Random Forest)
![Prediction Model](https://github.com/user-attachments/assets/3aa6b273-8c67-4181-945a-89acec90784f)
### 2. Analysis & Insights
![Analysis and Insights](https://github.com/user-attachments/assets/d447fc54-be55-4d94-88c3-98b3be162484)

## Technology stack
- **Frontend:** Streamlit
- **Backend:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, Numpy
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Data:** [data.gov.sg](https://data.gov.sg/collections/189/view)

## Data Processing
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

## References
- HDB resale data: https://data.gov.sg/collections/189/view
- HDB property information data: https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view
- Geocoding API: https://www.onemap.gov.sg/apidocs/search
- Procedures to buy a resale flat: https://www.hdb.gov.sg/business/estate-agents-and-salespersons/buying-a-resale-flat

## Contact
Darryl Lee - [LinkedIn](https://www.linkedin.com/in/darryl-lee-jk/)
