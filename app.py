import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# --- Feature Engineering Function (NEW) ---
def engineer_features(df):
    """Creates time-series features from a dataframe."""
    df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)
    
    # Lags
    lags = [7, 14, 28, 365]
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
        
    # Rolling Windows
    windows = [7, 28]
    for window in windows:
        df[f'sales_rolling_mean_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'sales_rolling_std_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).std()
        )
    
    df.fillna(0, inplace=True)
    return df
    
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
# --- Forecasting Logic ---
st.header(f'Sales Forecast for the Next {forecast_days} Days')

if st.button('Generate Forecast'):
    with st.spinner('Generating forecast... This may take a moment.'):
        # 1. Get historical data for the selected item/store
        historical_data = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()

        # 2. Create future dataframe
        last_date = historical_data['date'].max()
        future_dates = pd.to_datetime([last_date + timedelta(days=x) for x in range(1, forecast_days + 1)])
        future_df = pd.DataFrame({
            'date': future_dates,
            'store': store_id,
            'item': item_id,
            'sales': 0 # Placeholder, will not be used in features
        })

        # 3. Combine historical and future data
        combined_df = pd.concat([historical_data, future_df], ignore_index=True)
        
        # 4. Engineer features on the combined dataframe
        # This allows lags/rolling windows to be calculated correctly using historical data
        combined_df_features = engineer_features(combined_df)
        
        # 5. Create date features for the combined dataframe
        combined_df_features['year'] = combined_df_features['date'].dt.year
        combined_df_features['month'] = combined_df_features['date'].dt.month
        combined_df_features['day'] = combined_df_features['date'].dt.day
        combined_df_features['dayofweek'] = combined_df_features['date'].dt.dayofweek

        # 6. Select only the future rows for prediction
        prediction_df = combined_df_features[combined_df_features['date'] >= future_dates[0]]

        # 7. Define feature list (must match training)
        features_for_pred = [
            'store', 'item', 'year', 'month', 'day', 'dayofweek',
            'sales_lag_7', 'sales_lag_14', 'sales_lag_28', 'sales_lag_365',
            'sales_rolling_mean_7', 'sales_rolling_std_7',
            'sales_rolling_mean_28', 'sales_rolling_std_28'
        ]

        # 8. Make predictions
        predictions = model.predict(prediction_df[features_for_pred])

        # 9. Create and display forecast results
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Sales': np.round(predictions).astype(int)
        })
        
        st.line_chart(forecast_df.set_index('Date')['Forecasted Sales'])
        st.write("Forecast Data:")
        st.dataframe(forecast_df)