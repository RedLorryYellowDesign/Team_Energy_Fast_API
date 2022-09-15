# ---| ALL IMPORTS |---
from enum import Enum
from optparse import Option
from typing import List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel

# ---| ALL MODELS |---


class Calling_Forcast_Model(BaseModel):
    forecast_model: Optional[str]
        # # Submit Button
        # if st.button("Submit"):
        #     st.write("Model will be called here")
        #     name=User_Group_Selected,tariff=User_Tarrif
        #     # Joblib import model
        #     filename = f'model_{name}_{tariff}.joblib'
        #     m = joblib.load(filename)
        #     #print('model loaded succcessfully')
        #     # ---| MODEL WILL BE CALLE HERE |---
        #     # def model_call(User_Tarrif_Selected,User_Group_Selected): # model will be called here
        #     #     return Model_Result
        #     train_df, test_df = create_data(name = name, tariff = User_Tarrif)
        #     train_wd, test_wd = get_weather(train_df, test_df)
        #     # Calculate forecast and MAPE
        #     forecast = forecast_model(m=m, train_wd = train_wd, test_wd = test_wd, add_weather = True)
        #     #print('forecast made')
        #     mape = evaluate(test_df['KWH/hh'], forecast['yhat'])
        #     confidence=(1-mape)*100
        #     # Plot the graphs
        #     #print('now plotting')
        #     plot_graphs(test_df = test_df, forecast= forecast)
        #     #print('operation complete'
'''
class Gender (str, Enum):
    male = "male"
    female = "female"

class Role(str, Enum):
    admin = "admin"
    user = "user"
    student = "student"

class User(BaseModel):
    id: Optional[UUID] = uuid4()
    first_name: str
    last_name: str
    middle_name: Optional[str]
    gender: Gender
    roles: List[Role]

class User_Update_Requests(BaseModel):
    first_name: Optional[str]
    last_name: Optional[str]
    middle_name: Optional[str]
    roles: Optional[List[Role]]
'''
