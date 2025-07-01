import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Demand Forecaster",
    page_icon="📈",
    layout="wide"
)

# --- Load Model and Data ---
@st.cache_resource
def load_model():
    """Load the trained LightGBM model."""
    return joblib.load('lgbm_model.joblib')

@st.cache_data
def load_data():
    """Load and preprocess the data."""
    df = pd.read_csv('data/train.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

model = load_model()
df = load_data()

# --- App Title ---
st.title('📈 Interactive Sales Demand Forecasting')
st.markdown("This dashboard allows you to forecast sales demand for any store and item combination.")

# --- Sidebar for User Input ---
st.sidebar.header('Select Your Filters')
store_id = st.sidebar.selectbox('Select Store ID', df['store'].unique())
item_id = st.sidebar.selectbox('Select Item ID', df['item'].unique())
forecast_days = st.sidebar.slider('Number of Days to Forecast', 7, 90, 30) # Min, Max, Default

# --- Filter Data for Visualization ---
history_df = df[(df['store'] == store_id) & (df['item'] == item_id)]

# --- Main Panel ---
st.header(f'Historical Sales for Store {store_id}, Item {item_id}')
st.line_chart(history_df.set_index('date')['sales'])

# --- Forecasting Logic ---
st.header(f'Sales Forecast for the Next {forecast_days} Days')

# Create a future dataframe for predictions
last_date = history_df['date'].max()
future_dates = pd.to_datetime([last_date + timedelta(days=x) for x in range(1, forecast_days + 1)])

future_df = pd.DataFrame({
    'date': future_dates,
    'store': store_id,
    'item': item_id
})

# Engineer features for the future dataframe
future_df['year'] = future_df['date'].dt.year
future_df['month'] = future_df['date'].dt.month
future_df['day'] = future_df['date'].dt.day
future_df['dayofweek'] = future_df['date'].dt.dayofweek

# Make predictions
features_for_pred = ['store', 'item', 'year', 'month', 'day', 'dayofweek']
predictions = model.predict(future_df[features_for_pred])

# Create forecast results dataframe
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Sales': np.round(predictions).astype(int) # Round to nearest whole number
})

# Display forecast chart and data
st.line_chart(forecast_df.set_index('Date')['Forecasted Sales'])
st.write("Forecast Data:")
st.dataframe(forecast_df)