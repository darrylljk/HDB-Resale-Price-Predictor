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

# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

st.title("Exploratory Data Analysis (EDA)")

df = pd.read_csv("data-github/geocoded_data_with_hdb_property_info_processed.csv")

# -------------------------------------------------------------
# Data Visualization
# -------------------------------------------------------------
st.write("""
Here, we'll use charts to explore interactions between features, reveal underlying patterns, and extract valuable insights.
""")
# -------------------------------------------------------------
# [EDA] Singapore Map - Resale Flat Prices by Town
# -------------------------------------------------------------
map_fig = px.scatter_mapbox(
    df,
    lat='latitude',
    lon='longitude',
    color='resale_price', 
    opacity=0.5,  
    title='Resale Flat Prices across Singapore',
    hover_data={
        'town': True,
        'block': True,
        'latitude': False,   
        'longitude': False,  
        'resale_price': True,
        'flat_type': True,
        'year': True,
        'floor_area_sqm': True
    },
    color_continuous_scale='Oranges'
    # color_continuous_scale=[[0, 'white'], [1, '#004d40']]  
    # color_continuous_scale=[[0, 'white'], [1, 'royalblue']]
    # color_continuous_scale='Blues'
)

# set the marker opacity
map_fig.update_traces(marker=dict(opacity=0.1))

# set the map style and layout
map_fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=11,
    mapbox_center={"lat": df['latitude'].mean(), "lon": df['longitude'].mean()},
    showlegend=True,
    width=1200,
    height=800
)

# display the map in streamlit
st.plotly_chart(map_fig, use_container_width=True)
st.write("Predictably, HDB flats closer to the CBD consistently command higher prices, underscoring the enduring allure of central locations. These areas offer unparalleled accessibility to job hubs, amenities, and public transport—a premium that often eclipses considerations like space or the property’s age. The trend reflects a deep-seated preference for convenience and connectivity in urban living.")


# -------------------------------------------------------------
# [EDA] Bar charts - Distribution of flat types & flat model
# -------------------------------------------------------------
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Flat Type", "Flat Model"))

# flat type distribution
flat_type_count = df['flat_type'].value_counts().reset_index()
flat_type_count.columns = ['flat_type', 'count']  
flat_type_chart = px.bar(
    flat_type_count,
    x='flat_type',
    y='count',
    labels={'flat_type': 'Flat Type', 'count': 'Count'},
    color_discrete_sequence=['#f07167']
)

# flat model distribution
flat_model_count = df['flat_model'].value_counts().reset_index()
flat_model_count.columns = ['flat_model', 'count']  
flat_model_chart = px.bar(
    flat_model_count,
    x='flat_model',
    y='count',
    labels={'flat_model': 'Flat Model', 'count': 'Count'},
    color_discrete_sequence=['#00afb9']
    # color_discrete_sequence=['#636EFA'], 

)

# add traces to subplots
for trace in flat_type_chart.data:
    fig.add_trace(trace, row=1, col=1)
for trace in flat_model_chart.data:
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(
    title_text="Flat Type & Flat Model Distributions",
    title_font=dict(size=16, weight='bold'), 
    showlegend=False,
    height=500,
    width=1200,
)

fig.update_xaxes(row=1, col=1, tickangle=45)
fig.update_xaxes(row=1, col=2, tickangle=45)
fig.update_yaxes(title_text="Count")

st.plotly_chart(fig, use_container_width=True)
st.write("""
         Most resale flats are 3-, 4-, or 5-room units, predominantly categorized under “Model A,” “Improved,” “New Generation,” or “Premium Apartment” designs, reflecting the evolving housing needs of families across different eras. Notably, newer models like Premium Apartments cater to higher-income households, while earlier models such as Improved and New Generation flats capture Singapore’s transition toward modern, functional living spaces.
""")


# -------------------------------------------------------------
# [EDA] Binning of resale prices and visualizing with a histogram
# -------------------------------------------------------------
bins = np.arange(0, df['resale_price'].max() + 50000, 100000)
df['price_range'] = pd.cut(df['resale_price'], bins=bins, include_lowest=True)

hist_fig = px.histogram(
    df,
    x='resale_price',
    nbins=40,
    title="Resale Prices Distribution",
    labels={'resale_price': 'Resale Price'}, 
    color='flat_type',    color_discrete_sequence=px.colors.qualitative.Set2,  
    # color_discrete_sequence=['#636EFA'], 
    opacity=0.8
)

hist_fig.update_layout(
    xaxis=dict(title='Resale Price (SGD)', tickprefix='$', tickformat=','),
    yaxis=dict(title='Count'),
    bargap=0.1, 
    width=1200, 
    height=500,
    title_font=dict(size=16, weight='bold'), 
)

st.plotly_chart(hist_fig, use_container_width=True)
st.write("Distribution of resale prices shows a strong concentration between \$300K and \$700K. There is a noticeable right skew, with fewer flats priced beyond \$1M. The presence of flats below \$300K indicates affordability options, catering to smaller flat types or older properties in less central regions.")
st.write('')
st.write('')

# -------------------------------------------------------------
# [EDA] Boxplot - resale prices by town and region
# -------------------------------------------------------------
location_region_lookup = {
    'North': ['ANG MO KIO', 'SEMBAWANG', 'WOODLANDS', 'YISHUN'],
    'North-East': ['HOUGANG', 'PUNGGOL', 'SENGKANG', 'SERANGOON'],
    'East': ['BEDOK', 'MARINE PARADE', 'PASIR RIS', 'TAMPINES'],
    'West': ['BUKIT BATOK', 'BUKIT PANJANG', 'CHOA CHU KANG', 'CLEMENTI', 'JURONG EAST', 'JURONG WEST'],
    'Central': ['BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA', 'GEYLANG', 'KALLANG/WHAMPOA', 'QUEENSTOWN', 'TOA PAYOH']
}

fig = make_subplots(
    rows=1, 
    cols=len(location_region_lookup), 
    subplot_titles=list(location_region_lookup.keys())
)

region_colors = {
    'North': '#636EFA',
    'North-East': '#EF553B',
    'East': '#00CC96',
    'West': '#AB63FA',
    'Central': '#FFA15A'
}

for i, (region, towns) in enumerate(location_region_lookup.items(), start=1):
    region_data = df[df['town'].isin(towns)]
    box_fig = px.box(
        region_data,
        x='town',
        y='resale_price',
        color_discrete_sequence=[region_colors[region]], 
    )
    for trace in box_fig.data:
        fig.add_trace(trace, row=1, col=i)

fig.update_layout(
    title=dict(
        text="Resale Prices by Town and Region",
        font=dict(size=16, weight='bold'),
        y=1  
    ),
    showlegend=False,
    width=1500,
    height=600,
    margin=dict(t=70, l=50, r=50, b=50),  
    xaxis_title=None,  
    yaxis_title=None, 
    plot_bgcolor="rgba(240, 240, 240, 1)"  
)

for annotation in fig['layout']['annotations']:
    annotation['y'] += 0.05 
for i in range(1, len(location_region_lookup) + 1):
    fig.update_xaxes(row=1, col=i, tickangle=45)  
    if i > 1:  
        fig.update_yaxes(row=1, col=i, showticklabels=False)
    else:  
        fig.update_yaxes(row=1, col=i, range=[0, 2_000_000])
st.plotly_chart(fig, use_container_width=True)

st.write("Suburban towns like Woodlands, Yishun, and Sembawang in the North consistently demonstrate lower resale prices, making them attractive to cost-conscious buyers. In contrast, towns in the Central region, such as Toa Payoh and Bishan, command the highest median prices, with notable variability that reflects a mix of premium flats and older developments. The East and North-East regions exhibit competitive pricing, generally below Central levels, but still show occasional outliers, likely representing high-end or uniquely situated properties. The price spread across regions highlights the combined influence of factors such as location demand, flat size, age, and proximity to amenities and transportation hubs.")



# -------------------------------------------------------------
# [EDA] Time Series - Resale Flat Prices Trend
# -------------------------------------------------------------
# dip in 2020 due to covid-19
df_trend = df[['year', 'resale_price', 'flat_type']]
df_trend_avg = df_trend.groupby(['year', 'flat_type']).agg({'resale_price': 'mean'}).reset_index()

trend_fig = px.line(
    df_trend_avg,
    x='year',
    y='resale_price',
    color='flat_type',
    markers=True,
    line_shape='linear',
    title='Resale Price Trend',
    labels={'resale_price': 'Average Resale Price (SGD)', 'year': 'Year', 'flat_type': 'Flat Type'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)

trend_fig.update_layout(
    width=1200,
    height=600,
    legend_title=dict(font=dict(size=14), text='Flat Type'),
    xaxis=dict(tickangle=45, title_font=dict(size=14), tickfont=dict(size=12), gridcolor='lightgrey'),
    yaxis=dict(
        title='Average Resale Price (SGD)',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        tickformat=',',
        gridcolor='lightgrey',
        range=[0, None],
        tick0=0,
        dtick=50000
    ),
    title=dict(font=dict(size=16, weight='bold')),
    plot_bgcolor='white'
)

trend_fig.update_xaxes(
    showgrid=True,
    gridwidth=0.5,
    gridcolor="rgba(200, 200, 200, 0.3)" 
)
trend_fig.update_yaxes(
    showgrid=True,
    gridwidth=0.5,
    gridcolor="rgba(200, 200, 200, 0.3)"
)
st.plotly_chart(trend_fig, use_container_width=True)
st.write("Resale prices for HDB flats have demonstrated a natural upward trajectory, consistent with the broader trend of appreciating property values in Singapore. Even with this predictable growth, the dip in 2020 stands out—a clear result of the COVID-19 pandemic disrupting economic confidence and housing demand. Following this, the strong recovery from 2021 onward highlights the resilience of the market, particularly for larger flat types, which saw greater demand as priorities shifted toward space and comfort in the wake of the pandemic.")

# -------------------------------------------------------------
# [EDA] Time series - % YoY change
# -------------------------------------------------------------
df_trend = df[['year', 'resale_price', 'flat_type']]
df_trend_avg = df_trend.groupby(['year', 'flat_type']).agg({'resale_price': 'mean'}).reset_index()

df_trend_avg['pct_change'] = df_trend_avg.groupby('flat_type')['resale_price'].pct_change() * 100
df_trend_avg['pct_change'] = df_trend_avg['pct_change'].fillna(0)

y_min = df_trend_avg['pct_change'].min()
y_max = df_trend_avg['pct_change'].max()

pct_change_fig = px.line(
    df_trend_avg,
    x='year',
    y='pct_change',
    color='flat_type',
    markers=True,
    line_shape='linear',
    title='Resale Price % YoY Change',
    labels={'pct_change': 'YoY % Change in Avg Resale Price', 'year': 'Year', 'flat_type': 'Flat Type'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)
pct_change_fig.update_layout(
    width=1200,
    height=600,
    legend_title=dict(font=dict(size=14), text='Flat Type'),
    xaxis=dict(tickangle=45, title_font=dict(size=14), tickfont=dict(size=12), gridcolor='lightgrey'),
    yaxis=dict(
        title='YoY % Change in Avg Resale Price',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        tickformat=',.1f',
        gridcolor='lightgrey',
        range=[y_min, y_max],  
    ),
    title=dict(font=dict(size=16, weight='bold')),
    plot_bgcolor='white'
)

pct_change_fig.update_xaxes(
    showgrid=True,
    gridwidth=0.5,
    gridcolor="rgba(200, 200, 200, 0.3)"  
)
pct_change_fig.update_yaxes(
    showgrid=True,
    gridwidth=0.5,
    gridcolor="rgba(200, 200, 200, 0.3)"
)

st.plotly_chart(pct_change_fig, use_container_width=True)
st.write("The year-on-year price changes tell a compelling story of market volatility and resilience. While the steep decline in 2020 reflects the immediate impact of the pandemic, the dramatic rebound in 2021 signals a rapid correction driven by pent-up demand and market recovery. Interestingly, the fluctuations over the years vary by flat type, with executive flats showing the sharpest shifts, underscoring their sensitivity to changing market conditions and preferences. By 2024, the YoY changes reflect a market settling back into stability, balancing between growth and consolidation.")


# -------------------------------------------------------------
# [EDA] Scatterplot - Impact of Floor Area on Resale Price
# -------------------------------------------------------------
scatter_fig = px.scatter(
    df, 
    x='floor_area_sqm', 
    y='resale_price', 
    color='flat_type', 
    title='Floor Area vs Resale Price Segregated by Flat Type', 
    labels={'floor_area_sqm': 'Floor Area (sqm)', 'resale_price': 'Resale Price (SGD)'},
    color_discrete_sequence=px.colors.qualitative.Set2
)

scatter_fig.update_traces(marker=dict(opacity=0.5))

scatter_fig.update_layout(
    title={'text': 'Impact of Floor Area on Resale Price', 'font': {'size': 16, 'weight': 'bold'}},
    xaxis_title='Floor Area (sqm)',
    yaxis_title='Resale Price (SGD)',
    height=600,
    width=1200
)

scatter_fig.update_xaxes(range=[0, 250]) 
st.plotly_chart(scatter_fig, use_container_width=True)
st.write("""
In Singapore’s highly competitive HDB resale market, floor area emerges as a dominant factor in determining resale prices. The chart reveals a clear upward trend, where larger flats command significantly higher prices. For instance, executive flats and multi-generation units (typically above 110 sqm) are priced at a premium, reflecting their appeal to larger families seeking spacious living arrangements. On the other hand, smaller flats like 2-room and 3-room units cluster at lower price points, catering to singles, couples, or budget-conscious buyers.
""")
st.write("""
However, while absolute prices are important, the price per square meter (sqm) provides a more nuanced understanding of value. This metric, already calculated in our dataset, allows for more meaningful comparisons across flat types and locations, enabling buyers and policymakers to identify price disparities and assess affordability more effectively. By normalizing for size, the price per sqm highlights how some smaller flats can command disproportionately high premiums due to location or age, reflecting their desirability despite their limited space.      
""")
# -------------------------------------------------------------
# [EDA] Resale price by storey range
# -------------------------------------------------------------
fig = px.box(
    df,
    x='storey_range',
    y='resale_price',
    title='Impact of Storey Range on Resale Price',
    labels={'storey_range': 'Storey Range', 'resale_price': 'Resale Price (SGD)'},
    color_discrete_sequence=px.colors.qualitative.Set2,
    category_orders={'storey_range': sorted(df['storey_range'].unique())}  
)

fig.update_layout(
    height=600,
    width=900,
    title_font=dict(size=16, weight='bold'),
    xaxis_title='Storey Range',
    yaxis_title='Resale Price (SGD)',
    xaxis=dict(tickangle=45), 
)

st.plotly_chart(fig, use_container_width=True)
st.write("In Singapore, the floor level of a flat plays a nuanced role in its resale value. Flats on higher storey ranges, particularly those above 30 stories, tend to fetch higher prices, as buyers value better ventilation, unobstructed views, and reduced street noise. However, the chart also reflects significant overlap across storey ranges, indicating that factors such as location, flat type, and proximity to MRT stations or amenities often outweigh floor level alone. Notably, the broader spread of resale prices for higher floors suggests that premium units, such as those with unique features or located in prime neighborhoods, are disproportionately represented in these categories.")

# -------------------------------------------------------------
# [EDA] Resale price by remaining lease
# -------------------------------------------------------------
fig = px.scatter(
    df,
    x='remaining_total_years',
    y='resale_price',
    title='Impact of Remaining Property Lease (Years) on Resale Price',
    labels={'remaining_years': 'Remaining Property Lease (Years)', 'resale_price': 'Resale Price (SGD)'},
    opacity=0.2, color='flat_type',
    # color_discrete_sequence=['#636EFA'] 
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig.update_layout(
    height=600,
    width=900,
    title_font=dict(size=16, weight='bold'),
    xaxis_title='Remaining Lease Years',
    yaxis_title='Resale Price (SGD)',
)

st.plotly_chart(fig, use_container_width=True)
st.write("In Singapore’s HDB market, the remaining lease years have a moderate influence on resale prices, reflecting the leasehold nature of HDB flats. As expected, flats with shorter remaining leases (below 50 years) exhibit noticeably lower prices, aligning with the depreciation of leasehold properties as they approach the end of their tenure. Buyers often perceive shorter leases as less desirable due to the diminished value and restrictions on financing and CPF usage.")

# -------------------------------------------------------------
# [EDA] Violin plot - Impact of location on Price per sqm
# -------------------------------------------------------------
fig = px.violin(
    df,
    x='Region',  
    y='price_per_sqm',  
    color='Region',  
    box=True,  
    points=None,  
    title='Impact of Location on Price per sqm',
    labels={'Region': 'Region', 'resale_price': 'Price per sqm (SGD)'},
    color_discrete_sequence=px.colors.qualitative.Set2,  
)

fig.update_layout(
    height=600,
    width=900,
    title_font=dict(size=16, weight='bold'),
    xaxis_title='Region',
    yaxis_title='Price per sqm (SGD)',
    legend_title=dict(text='Region'),
)

st.plotly_chart(fig, use_container_width=True)
st.write("The Central region commands the highest price per sqm, with a wide spread indicating both premium properties and variability in desirability. This reflects its proximity to the Central Business District (CBD) and extensive access to key amenities, public transport, and high-demand neighborhoods like Toa Payoh and Bishan. Conversely, the North and North-East regions demonstrate more affordable price per sqm, appealing to cost-conscious buyers seeking spacious living in quieter neighborhoods like Yishun, Woodlands, and Sengkang. The East and West regions strike a middle ground, reflecting their balance of accessibility and livability, with mature estates like Bedok and Clementi standing out. These regional price disparities reinforce how location remains a key determinant of value, even when normalized by flat size.")

# -------------------------------------------------------------
# [EDA] Correlation Matrix of Numerical Features
# -------------------------------------------------------------
# filter to keep only numerical columns
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

corr_fig = px.imshow(
    corr_matrix,
    text_auto='.3f',
    color_continuous_scale='RdBu',
    zmin=-1, 
    zmax=1,
    title='Correlation Matrix for Numerical Features'
)

corr_fig.update_layout(
    width=1200, 
    height=1000,
    coloraxis_colorbar=dict(title="Correlation")  
)

st.plotly_chart(corr_fig, use_container_width=True)
st.write('Numerical features that are most correlated to `resale_price` are floor area, remaining duration of flat, max floor level (building height), and year_completed (building age). Features that have moderate to strong correlation with resale price will be considered and prioritized when building the machine learning model.')

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



# -------------------------------------------------------------
# Unused charts
# -------------------------------------------------------------

# -------------------------------------------------------------
# [EDA] Boxplot - Distribution of resale price by town
# -------------------------------------------------------------
# box_fig = px.box(
#     df,
#     x='town',
#     y='resale_price',
#     color='town',
#     title='Resale Prices by Town',
#     labels={'resale_price': 'Resale Price (SGD)', 'town': 'Town'},
#     color_discrete_sequence=px.colors.qualitative.Set2
# )
# box_fig.update_layout(
#     title_font=dict(size=16, weight='bold'),
#     xaxis_title='Town',
#     yaxis_title='Resale Price (SGD)',
#     xaxis=dict(tickangle=60),  
#     showlegend=False,  
#     width=1200,  
#     height=600
# )
# st.plotly_chart(box_fig, use_container_width=True)
# st.write('insert comments...')
