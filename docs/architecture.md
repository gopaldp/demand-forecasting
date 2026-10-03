# Architecture Overview

This document describes the modular architecture, data flow, and core components of the **Sales Demand Forecasting** application.

## System Components

The project consists of two primary applications, a pre-trained machine learning model, historical datasets, and containerization configurations:

1. **Main Interactive Dashboard (`app.py`)**:
   - Built with **Streamlit**.
   - Provides user controls in the sidebar for selecting a Store ID, Item ID, and forecast horizon (7 to 90 days).
   - Loads historical sales data from `data/train.csv`.
   - Renders historical sales charts and multi-step demand forecasts using the pre-trained LightGBM model.

2. **Feature Engineering Pipeline (`engineer_features`)**:
   - Located within `app.py`.
   - Sorts time-series records by `[store, item, date]`.
   - Computes lagging features: `sales_lag_7`, `sales_lag_14`, `sales_lag_28`, and `sales_lag_365`.
   - Computes rolling window statistics (mean and standard deviation) over 7-day and 28-day windows.
   - Extracts date components (`year`, `month`, `day`, `dayofweek`) for temporal modeling.

3. **Model Inspector (`streamlit_app.py`)**:
   - A secondary Streamlit application.
   - Loads `lgbm_model.joblib`.
   - Inspects the model's feature schema (`n_features_in_`, booster feature names, or fallbacks).
   - Allows users to test predictions via custom JSON payloads or manual numeric input fields.

4. **Pre-trained Model (`lgbm_model.joblib`)**:
   - A serialized **LightGBM** gradient boosted decision tree model trained on historical sales data.

5. **Data Layer (`data/`)**:
   - `data/train.csv`: Historical training data containing `date`, `store`, `item`, and `sales`.
   - `data/test.csv`: Test dataset for evaluation.

## Data & Control Flow

1. **User Request**: The user selects a store and item in the Streamlit UI (`app.py`) and specifies the forecast horizon $N$ (7–90 days).
2. **Historical Context Retrieval**: The app filters `data/train.csv` for the selected store and item.
3. **Future Frame Construction**: Appends $N$ future date rows with placeholder sales values.
4. **Feature Extraction**: `engineer_features()` processes the combined historical and future DataFrame to dynamically generate lag and rolling features.
5. **Inference**: The LightGBM model (`lgbm_model.joblib`) evaluates the feature matrix for future dates, outputting rounded integer sales predictions.
6. **Visualization**: Streamlit renders interactive line charts and tabular summaries of the forecasted demand.
