# Fast API for the main app
# ---| ALL IMPORTS |---
from typing import List
from fastapi import FastAPI, HTTPException
from app.Team_Energy.RNN_predict import forecast_model, evaluate, plot_graphs
from app.Team_Energy.predict import get_holidays, get_weather, create_data
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

@app.get("/model/predict")
async def predict_Model(name, tariff):
    # Joblib import model
    filename = f'app/Team_Energy/Prophet_models/model_{name}_{tariff}.joblib'
    m = joblib.load(filename)
    train_df, test_df,val_df = create_data(name, tariff)
    train_df, test_df = create_data(name, tariff)
    train_wd, test_wd = get_weather(train_df, test_df)
    # Calculate forecast and MAPE
    forecast = forecast_model(m=m, train_wd = train_wd, test_wd = test_wd, add_weather = True)
    mape = evaluate(test_df['KWH/hh'], forecast['yhat'])

    forecast_list = forecast.tolist()
    return {'prediction': [forecast_list], 'accuracy': mape }




# ef forecast_model(m,train_wd,test_wd,add_weather=True):
#     future = m.make_future_dataframe(periods=48*27+1, freq='30T')
#     if add_weather==True:
#         wd_filt_future=future[['ds']].merge(pd.concat([future,pd.concat([train_wd,test_wd],axis=0)]),left_on='ds',right_on='DateTime',how='inner').drop(columns='DateTime')
#         temp_future=wd_filt_future['temperature'].interpolate(method='linear')
#         future['temp']=temp_future
#         fcst = m.predict(future)
#     else:
#         fcst = m.predict(future)
#     forecast=fcst.loc[fcst['ds']>='2014-02-01 00:00:00',['ds','yhat']]
#     return forecast

# Evaluate model
def evaluate(actual,forecast):
    return np.round(mean_absolute_percentage_error(actual,forecast),4)

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

    # convert numpy array to list
    predicted_consumption_list = predicted_consumption.tolist()

    # return {"Predict":predicted_consumption_JSON,"acuracy":mape}
    return {'prediction': [predicted_consumption_list], 'accuracy': mape }
