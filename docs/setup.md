# Setup Guide

This guide details the local setup, Docker containerization, Dev Container configuration, and troubleshooting for the **Sales Demand Forecasting** project.

## Local Development Setup

### Prerequisites
- Python 3.9 or Python 3.11
- `pip` package manager

### Steps
1. Clone the repository and navigate into the project directory:
   ```bash
   git clone <repository-url>
   cd demand-forecasting
   ```

2. (Optional) Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the main Streamlit application:
   ```bash
   streamlit run app.py
   ```
   The dashboard will be available at `http://localhost:8501`.

5. Alternatively, run the model inspector application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Docker Setup

The project includes a `Dockerfile` based on Python 3.9 that installs system-level dependencies (`libgomp1` required by LightGBM), installs requirements, and exposes Streamlit on port 8501.

### Building and Running the Docker Container
1. Build the Docker image:
   ```bash
   docker build -t demand-forecasting .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 demand-forecasting
   ```
3. Open `http://localhost:8501` in your browser.

## Dev Container Setup

The repository includes a `.devcontainer/devcontainer.json` configuration for Python 3.11 environments (such as VS Code Remote - Containers / GitHub Codespaces). Opening the repository in the dev container automatically provisions the required development toolchain.

## Troubleshooting

- **LightGBM runtime error (`libgomp.so.1: cannot open shared object file`)**:
  Ensure `libgomp1` is installed on your Linux distribution (handled automatically in the `Dockerfile` via `apt-get install -y libgomp1`).
- **Streamlit port conflict (`Port 8501 is already in use`)**:
  Run Streamlit on a different port using:
  ```bash
  streamlit run app.py --server.port 8502
  ```
