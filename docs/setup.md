# Local Setup & Troubleshooting

This guide covers setting up the development environment, running the application locally, using Docker, and resolving common issues.

## Local Environment Setup

### Prerequisites
- Python 3.9 or higher
- `pip` package manager

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/gopaldp/demand-forecasting.git
   cd demand-forecasting
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### 1. Main Application
Run the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your web browser and navigate to `http://localhost:8501`.

### 2. Model Inspection Utility
Run the secondary inspection app:
```bash
streamlit run streamlit_app.py
```

## Docker Setup

The application includes a `Dockerfile` based on `python:3.9-slim` with `libgomp1` installed for LightGBM support.

1. Build the Docker image:
   ```bash
   docker build -t demand-forecasting .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 demand-forecasting
   ```

## Dev Containers

For Visual Studio Code or GitHub Codespaces users, `.devcontainer/devcontainer.json` provides a pre-configured Python 3.11 environment with automatic package installation and port forwarding for port 8501.

## Troubleshooting

- **Missing `libgomp1` Error (Linux)**: LightGBM requires OpenMP (`libgomp1`). Ensure it is installed via your package manager (e.g., `sudo apt-get install libgomp1`) or use the provided Dockerfile.
- **Port 8501 In Use**: If port 8501 is already occupied by another Streamlit process, run Streamlit on an alternate port:
  ```bash
  streamlit run app.py --server.port 8502
  ```
