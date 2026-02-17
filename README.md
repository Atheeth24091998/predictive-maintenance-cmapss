# Predictive Maintenance: Remaining Useful Life (RUL) Prediction (NASA C-MAPSS)

This project predicts the **Remaining Useful Life (RUL)** of turbofan engines using the **NASA C-MAPSS** dataset.  
In simple words: given an engine’s sensor readings over time, the model estimates **how many cycles are left before the engine fails**.

I built this project to practice real-world machine learning and MLOps concepts. 

---

## What this project does

- Loads and cleans raw sensor data from the NASA C-MAPSS dataset  
- Creates RUL labels for each engine and cycle  
- Applies feature scaling and removes constant features  
- Builds time-series features using sliding windows  
- Trains multiple models:
  - Random Forest  
  - XGBoost  
  - LSTM (PyTorch)  
- Evaluates models using RMSE and MAE  
- Serves predictions through a FastAPI endpoint  
- Packages the API using Docker for easy deployment  

---

## Dataset

The NASA C-MAPSS dataset contains simulated run-to-failure data for turbofan engines.

Each engine:
- Runs from healthy state until failure  
- Has multiple sensor readings per operating cycle  
- The task is to predict **how many cycles remain before failure (RUL)**  

Official source:  
https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

---

## Project structure

```text
predictive-maintenance-cmapss/
├── data/
│   └── raw/                  # train/test/RUL files
├── models/                   # saved models (e.g., xgb_model.pkl)
├── notebooks/                # exploration notebooks
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── pipeline.py
│   └── train_model.py
├── api/
│   └── main.py               # FastAPI app
├── requirements.api.txt     # minimal dependencies for API
├── Dockerfile
└── README.md
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt

```
## Run the API locally (FastAPI)

### Start the API server:

```bash
uvicorn api.main:app --reload

```

## Example request

**POST** `/predict`

```bash
{
  "data": []
}
```


## Example response
```bash
{
  "predicted_rul": 117.15
}
```


## Run with Docker

### Build the Docker image:
```bash
docker build -t rul-api .
``` 

### Run the container:
```bash
docker run -p 8000:8000 rul-api
```



## Tools & Concepts Learned

**Tools:** 
Python, Pandas, NumPy, Scikit-learn, XGBoost, PyTorch, FastAPI, Docker, MLflow, Git  
**Concepts:** 
Time-series feature engineering, Remaining Useful Life (RUL) prediction, regression models, model evaluation (RMSE/MAE), Fast API-based model serving, docker containerization, reproducible ML pipelines
