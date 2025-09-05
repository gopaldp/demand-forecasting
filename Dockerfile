# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (helpful for some scientific wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest (model, app, notebooks if needed)
COPY lgbm_model.joblib /app/lgbm_model.joblib
COPY streamlit_app.py /app/app.py
# If you need notebooks inside the image, uncomment:
# COPY notebooks /app/notebooks

# Streamlit runs on 8501 by default
EXPOSE 8501

# For Streamlit in container (no browser, friendly logs)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false

# Run the app (adjust if you use a different entrypoint)
CMD ["streamlit", "run", "app.py"]
