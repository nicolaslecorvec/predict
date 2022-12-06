from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from gravitw.logic.normalization import Dataset
import torch
from gravitw.logic.evaluate import evaluate
import re
from interface.import_model import import_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict():      # 1
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """
    id_ = [f"0dc4c8ed0"] #Create a label.csv   PG is id's name. the num in range is the id
    label_ = pd.DataFrame(data = id_, columns = ["id"])
    label_['target']=0.5
    label_.to_csv("label_.csv")


    model = torch.jit.load('/home/nicolas/code/nicolaslecorvec/predict/model_scripted.pt')
    model.eval()

    #make the prediction
    dataset_test = Dataset("/home/nicolas/code/nicolaslecorvec/predict/0dc4c8ed0.hdf5", "label_.csv")
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, num_workers=2, pin_memory=True)

    test = evaluate(model, loader_test, compute_score=False)
    return print(test)

    #return dict({'fare_amount': float(y_pred[0][0])})


@app.get("/")
def root():
    return     {
    'greeting': 'Hello'
    }
