# Sales Demand Forecasting

An interactive sales demand forecasting application powered by LightGBM and Streamlit. The application allows users to select a store, an item, and a forecast horizon (7–90 days) to visualize historical sales data and generate future sales predictions based on robust time-series feature engineering.

## Key Features

- **Interactive Streamlit Dashboard (`app.py`)**: Main application offering store and item filtering, historical sales visualization, and custom forecast horizons (7 to 90 days).
- **Alternative Inspection App (`streamlit_app.py`)**: Simplified utility app for inspecting the trained LightGBM model feature schema and making direct predictions with custom numerical inputs or JSON payloads.
- **Automated Feature Engineering**: Computes rigorous time-series features on-the-fly including sales lags (7, 14, 28, 365 days) and rolling window statistics (mean and standard deviation over 7 and 28 days) per store and item.
- **Pre-trained LightGBM Model**: Utilizes a serialized LightGBM regression model (`lgbm_model.joblib`) trained on historical sales datasets.

## Tech Stack

- **Python** (3.9 / 3.11)
- **Data Manipulation & Analysis**: `pandas`, `NumPy`
- **Machine Learning**: `lightgbm`, `scikit-learn`, `joblib`
- **Interactive UI**: `streamlit`
- **Environment & Deployment**: Docker (`Dockerfile`), Dev Containers (`.devcontainer/devcontainer.json`)

## Prerequisites

- Python 3.9+
- pip
- System libraries: `libgomp1` (required for LightGBM on Linux environments)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gopaldp/demand-forecasting.git
   cd demand-forecasting
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Main Streamlit App

To launch the main interactive sales demand forecaster:
```bash
streamlit run app.py
```
The app will be accessible at `http://localhost:8501`.

### Running the Model Inspector App

To launch the alternative feature schema inspection app:
```bash
streamlit run streamlit_app.py
```

## Configuration

This project does not require environment variables or external configuration files. All inputs are handled interactively through the Streamlit UI.

## Project Structure

```text
demand-forecasting/
├── app.py                # Main interactive Streamlit application
├── streamlit_app.py      # Alternative model inspection and prediction app
├── lgbm_model.joblib     # Pre-trained LightGBM model binary
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image configuration (Python 3.9-slim)
├── .devcontainer/        # Dev Container configuration
│   └── devcontainer.json
├── data/                 # Training and test datasets
│   ├── train.csv         # Historical sales data (date, store, item, sales)
│   └── test.csv
└── notebooks/            # Jupyter notebooks for EDA and modeling
    └── 1-EDA-and-Modeling.ipynb
```

## Testing

Currently, automated test suites are not configured in this repository. Model training and exploration workflows are documented in `notebooks/1-EDA-and-Modeling.ipynb`.

## Docker & Deployment

### Building and Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t demand-forecasting .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 demand-forecasting
   ```

3. Open your browser and navigate to `http://localhost:8501`.

## Links to Documentation

- [Architecture (`docs/architecture.md`)](docs/architecture.md)
- [Local Setup & Troubleshooting (`docs/setup.md`)](docs/setup.md)
