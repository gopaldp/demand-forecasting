# Sales Demand Forecasting

An interactive sales demand forecasting application built with Python, LightGBM, and Streamlit. It allows users to view historical sales and generate multi-day demand forecasts for any store and item combination.

## Key Features

- **Interactive Streamlit Dashboard (`app.py`)**: Select store IDs, item IDs, and forecast horizons (7 to 90 days) to visualize historical sales trends and generated forecasts.
- **Advanced Feature Engineering**: Computes time-series lags (7, 14, 28, and 365 days) and rolling window statistics (mean and standard deviation over 7 and 28 days) per store and item.
- **LightGBM Model**: Uses a pre-trained gradient boosting model (`lgbm_model.joblib`) for fast and accurate demand predictions.
- **Model Inspector App (`streamlit_app.py`)**: A secondary interface for inspecting the underlying model's feature schema and making direct predictions.
- **Containerization & Deployment Support**: Fully configured with a `Dockerfile` and Dev Container setup for seamless containerized execution.

## Tech Stack

- **Python** (3.9 / 3.11)
- **Data Manipulation & Scientific Computing**: `pandas`, `NumPy`
- **Machine Learning**: `scikit-learn`, `lightgbm`, `joblib`
- **Web UI**: `Streamlit`
- **Environment & Deployment**: Docker, Dev Containers

## Prerequisites

- Python 3.9 or higher
- `pip` package manager

## Installation

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone <repository-url>
   cd demand-forecasting
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Main Forecasting App
Run the primary interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to interact with the dashboard, choose stores/items, adjust the forecast horizon, and generate demand forecasts.

### Model Inspector App
To inspect model feature schemas and test individual feature vectors via JSON or manual inputs:
```bash
streamlit run streamlit_app.py
```

## Configuration

The application operates out-of-the-box using the provided data and pre-trained model binaries. No external environment variables or `.env` files are required for local execution.

## Project Structure

```text
demand-forecasting/
├── app.py                  # Main Streamlit forecasting dashboard
├── streamlit_app.py        # Secondary model feature inspection app
-─ lgbm_model.joblib        # Pre-trained LightGBM model binary
├── requirements.txt        # Python package dependencies
├── Dockerfile              # Docker container configuration
├── data/
│   ├── train.csv           # Historical training data (date, store, item, sales)
│   └── test.csv            # Test dataset
├── notebooks/
│   └── 1-EDA-and-Modeling.ipynb # Exploratory data analysis and model training notebook
└── .devcontainer/
    └── devcontainer.json   # VS Code Dev Container configuration
```

## Testing

*Note: Automated test suites are not currently implemented in this repository. Model validation and exploratory analysis are documented in `notebooks/1-EDA-and-Modeling.ipynb`.*

## Docker & Deployment

To build and run the application via Docker:

1. Build the Docker image:
   ```bash
   docker build -t demand-forecasting .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 demand-forecasting
   ```
   Access the app at `http://localhost:8501`.

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Setup Guide](docs/setup.md)
