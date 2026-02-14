# Remaining Useful Life (RUL) Prediction for Turbofan Engines

This project predicts the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset.

## Goal

The goal is to build an end-to-end machine learning pipeline:
- Load and clean the sensor data
- Create features from time series (window-based)
- Train simple models (Random Forest, XGBoost)
- Evaluate the models using RMSE and MAE
- Later: add deep learning models, experiment tracking, and deployment

I am doing this project to learn:
- How to work with time series degradation data
- How to design a clean ML project structure
- How to move towards MLOps (MLflow, FastAPI, Docker, Streamlit)

## Dataset

I use the NASA C-MAPSS turbofan engine degradation dataset.

Official source:
https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

The dataset was released by the NASA Prognostics Center of Excellence (PCoE).
It contains simulated run-to-failure sensor data of turbofan engines under different operating conditions.

Each engine unit:
- Operates until failure
- Has multiple sensor readings per cycle
- Includes different fault modes (depending on subset: FD001–FD004)

The task is: **predict how many cycles are left before the engine fails (RUL).**

## Project structure

```text
rul-turbofan-cmapss/
├── data/
│   ├── raw/          # original CMAPSS files
│   └── processed/    # cleaned and feature-engineered data
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate.py
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
