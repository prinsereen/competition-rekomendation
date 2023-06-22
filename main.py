from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    seni: int
    science: int
    sastra: int
    sosial: int
    bahasa: int

# Load the trained linear regression model
rekomendation_model = pickle.load(open('rekomendasi_kompetisi_dt.pkl', 'rb'))

# Create the FastAPI app
app = FastAPI()

# Define the API endpoint for prediction
# Define the API endpoint for prediction
@app.post('/rekomendation')
def make_prediction(input_data: ModelInput):
    # Extract the input features
    input_features = [input_data.seni, input_data.science, input_data.sastra, input_data.sosial, input_data.bahasa]

    # Make the prediction
    prediction = rekomendation_model.predict([input_features])

    # Return the prediction as a response
    return {'prediction': int(prediction[0])}



