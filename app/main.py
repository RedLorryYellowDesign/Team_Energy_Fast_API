# Fast API for the main app
# ---| ALL IMPORTS |---
from typing import List
from fastapi import FastAPI, HTTPException
from app.Team_Energy.RNN_predict import forecast_model, evaluate, plot_graphs
from app.Team_Energy.predict import forecast_model, get_holidays, get_weather, create_data
from app.Team_Energy.data import create_data
from app.Team_Energy.prepare import prepare_sequences
import joblib
import json
import numpy as np
import pandas as pd

app = FastAPI()

# ---| API END POINT |---
# These are all the API Keys that can be called.
@app.get("/")
def root():
    return {"message": "Hello World. This is the Team Energy API. Please use the API keys below to call the API."}

# ---| Calls to see of API is working |---
@app.get("/all_good")
def check():
    return {"API Up and Running"}

@app.get("/docs")
def docs_call():
    return {"message":"Hi, Docs to be added"}

# @app.get("/model/predict")
# async def predict_Model(name, tariff, add_weather=True):
#     # Joblib import model
#     filename = f'app/Team_Energy/Prophet_models/model_{name}_{tariff}.joblib'
#     m = joblib.load(filename)
#     train_df, test_df = create_data(name = name, tariff = tariff)
#     train_wd, test_wd = get_weather(train_df, test_df)
#     # Calculate forecast and MAPE
#     forecast = forecast_model(m=m, train_wd = train_wd, test_wd = test_wd, add_weather = True)
#     mape = evaluate(test_df['KWH/hh'], forecast['yhat'])
#     predicted_consumption_list = forecast.tolist()
#     # Evaluate model
#     # np.round(mape(test_set,predicted_consumption),4)
#     return {'prediction': [predicted_consumption_list],"test_df":[test_df],"test_wd":[test_wd] ,'accuracy': mape }

@app.get("/model/RNN_predict")
async def RNN_Model(name, tariff):
    # Joblib import model
    filename = f'app/Team_Energy/RNN/RNNmodel_{name}_{tariff}.joblib'
    m = joblib.load(filename)
    train_df, test_df,val_df = create_data(name, tariff)
    X_train, y_train, X_test, sc, test_set = prepare_sequences(train_df, test_df,val_df)
    # Calculate forecast and MAPE
    predicted_consumption = forecast_model(m,X_test,sc)
    mape = evaluate(test_set,predicted_consumption)
    test_set = test_set.tolist()
    predicted_consumption_list = predicted_consumption.tolist()
    # Evaluate model
    mape = evaluate(test_set,predicted_consumption)
    np.round(mape(test_set,predicted_consumption),4)
    return {'prediction': [predicted_consumption_list], "test" :[test_set],'accuracy': mape }


@app.get("/model/RNN_predict_test")
async def test_RNN(name, tariff):
    # joblib import model
    filename = f'RNNmodel_{name}_{tariff}.joblib'
    m = joblib.load(filename)
    train_df, test_df,val_df = create_data(name, tariff)
    X_train, y_train, X_test, sc, test_set = prepare_sequences(train_df, test_df,val_df)
    # Calculate forecast and MAPE
    predicted_consumption = forecast_model(m,X_test,sc)
    mape = evaluate(test_set,predicted_consumption)
    acuracy_round = np.round(mape(test_set,predicted_consumption),4)
    # code from RNN_Predict above
    # ---
    # converting to list
    test_set = test_set.tolist()
    predicted_consumption_list = predicted_consumption.tolist()
    # producting API json return
    return {'prediction': [predicted_consumption_list], "test" :[test_set],'accuracy': mape, "acuracy_round+" : acuracy_round}

@app.get("/model/predict_test")
async def test_predict(name, tariff):
    # Joblib import model
    filename = f'model_{name}_{tariff}.joblib'
    m = joblib.load(filename)

    train_df, test_df = create_data(name = name, tariff = tariff)
    train_wd, test_wd = get_weather(train_df, test_df)
    # Calculate forecast and MAPE
    forecast = forecast_model(m=m, train_wd = train_wd, test_wd = test_wd, add_weather = True)
    mape = evaluate(test_df['KWH/hh'], forecast['yhat'])
    # code from Predict above
    # ---
    # converting to list
    test_set = test_set.tolist()
    predicted_consumption_list = forecast.tolist()
    acuracy_round = np.round(mape(test_set,forecast),4)
    # producting API json return
    return {'prediction': [predicted_consumption_list], "test" :[test_set],'accuracy': mape, "acuracy_round+" : acuracy_round}
