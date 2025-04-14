from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from fastapi.responses import RedirectResponse

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'digital_fatigue_model.joblib')
model = joblib.load(model_path)

app = FastAPI(title="Digital Fatigue API")

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

# Request format
class FatigueInput(BaseModel):
    user_profile: str
    screen_time_hours: float
    nighttime_use: int
    app_switches: int
    social_media_ratio: float
    unlocks: int

# Endpoint
@app.post("/predict-fatigue")
def predict_fatigue(data: FatigueInput):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame([{
        'profile': data.user_profile,
        'screen_time_hours': data.screen_time_hours,
        'nighttime_use': data.nighttime_use,
        'app_switches': data.app_switches,
        'social_media_ratio': data.social_media_ratio,
        'unlocks': data.unlocks
    }])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return {"fatigue_level": prediction}
