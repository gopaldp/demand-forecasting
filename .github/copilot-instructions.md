# Copilot instructions

## Project

Sales demand forecasting with LightGBM, served as an interactive Streamlit
app. The user picks a store, an item and a forecast horizon (7–90 days);
the app shows historical sales and the forecast.

Stack: Python, pandas, NumPy, scikit-learn, LightGBM, Streamlit, joblib.
Dependencies are in `requirements.txt`.

How it runs:

1. `pip install -r requirements.txt`
2. `streamlit run app.py` (port 8501). `app.py` loads `lgbm_model.joblib` and
   `data/train.csv`, and builds features in `engineer_features()`: sales lags
   of 7/14/28/365 days, plus rolling mean and standard deviation over 7 and
   28 days per store and item.
3. `streamlit_app.py` is a second, simpler app that loads only the model and
   inspects its feature schema. Describe it from its code. `app.py` is the main
   app (the Dockerfile and dev container run `app.py`).
4. Training and exploration: `notebooks/1-EDA-and-Modeling.ipynb`. Use it as
   context only; there is no training script.
5. Docker: `Dockerfile` (Python 3.9, installs `libgomp1` for LightGBM,
   runs `streamlit run app.py`, port 8501). Dev container:
   `.devcontainer/devcontainer.json` (Python 3.11).
6. CI: `.github/workflows/main.yml` builds the Docker image and pushes it to
   Docker Hub on push to `main`. It uses the `DOCKERHUB_USERNAME` and
   `DOCKERHUB_TOKEN` secrets.

## Documentation notes

- There is **no `README.md`**. Create one.
- Data files: `data/train.csv` (≈17 MB) and `data/test.csv`. Describe their
  columns only from what the code reads (`date`, `store`, `item`, `sales`).
- Don't claim accuracy figures. None are computed outside the notebook.

## Ignore

- `venv/` (a committed virtual environment, very large),
  `.ipynb_checkpoints/`, and the binary `lgbm_model.joblib` (mention its
  purpose only).

## Conventions

- Keep changes small and focused. One concern per pull request.
- Base documentation on the actual code. Never invent features, metrics or
  commands. Mark anything uncertain with `TODO: confirm …`.
- Use UTF-8 for all text files.
