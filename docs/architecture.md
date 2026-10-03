# Architecture

The **Sales Demand Forecasting** application is structured around a modular Python and Streamlit architecture that integrates data loading, dynamic time-series feature engineering, machine learning inference, and web-based visualization.

## Component Overview

### 1. Main Interactive Application (`app.py`)
The primary user-facing interface built with Streamlit. It orchestrates the full forecasting pipeline:
- **Data Caching & Loading**: Loads historical sales data (`data/train.csv`) and the serialized LightGBM model (`lgbm_model.joblib`) utilizing Streamlit caching (`@st.cache_data` and `@st.cache_resource`).
- **Sidebar Controls**: Allows users to select specific store IDs, item IDs, and forecast horizons ranging from 7 to 90 days.
- **Historical Visualization**: Renders historical sales trends using Streamlit line charts.
- **Dynamic Feature Engineering**: Extends historical time series with future placeholder dates, computes time-series lags and rolling window statistics across combined datasets, extracts calendar attributes, and executes model inference.
- **Forecast Display**: Renders forecasted sales as interactive line charts and tabulated dataframes.

### 2. Model Inspection Utility (`streamlit_app.py`)
An alternative auxiliary Streamlit application designed for model inspection and validation:
- Extracts model feature schema dynamically (`n_features_in_`, `feature_name_`, or booster feature names).
- Provides manual input fields or raw JSON payload input for custom feature vector prediction.

### 3. Machine Learning Model (`lgbm_model.joblib`)
A pre-trained LightGBM regression model that expects a 14-feature input vector:
- Identifiers and calendar features: `store`, `item`, `year`, `month`, `day`, `dayofweek`
- Lag features: `sales_lag_7`, `sales_lag_14`, `sales_lag_28`, `sales_lag_365`
- Rolling window features: `sales_rolling_mean_7`, `sales_rolling_std_7`, `sales_rolling_mean_28`, `sales_rolling_std_28`

### 4. Data Layer (`data/`)
- `train.csv`: Historical training dataset containing columns `date`, `store`, `item`, and `sales`.
- `test.csv`: Evaluation dataset.

### 5. Training & Exploration (`notebooks/1-EDA-and-Modeling.ipynb`)
Jupyter notebook containing exploratory data analysis (EDA), feature engineering validation, and model training logic used to produce `lgbm_model.joblib`.

## Data and Control Flow

```text
[data/train.csv] ──> load_data() ──> Filter by Store & Item ──> Historical Display
                                                                      │
[lgbm_model.joblib] ──> load_model()                                  ▼
                                            Combine Historical + Future Dates
                                                                      │
                                                                      ▼
                                                       engineer_features(combined_df)
                                                                      │
                                                                      ▼
                                                       Extract Calendar Attributes
                                                                      │
                                                                      ▼
                                                           model.predict(features)
                                                                      │
                                                                      ▼
                                                         Forecast Line Chart & Table
```
