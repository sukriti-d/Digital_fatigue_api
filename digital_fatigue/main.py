from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from fastapi.responses import RedirectResponse
import logging
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'digital_fatigue_model.joblib')
try:
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

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
    try:
        logger.info(f"Received input data: {data}")
        
        # Create a DataFrame with the input data in the correct column order
        input_data = pd.DataFrame([{
            'screen_time_hours': data.screen_time_hours,
            'nighttime_use': data.nighttime_use,
            'app_switches': data.app_switches,
            'social_media_ratio': data.social_media_ratio,
            'unlocks': data.unlocks,
            'profile': data.user_profile
        }])
        
        logger.info(f"Created DataFrame: {input_data}")
        
        # Ensure columns are in the correct order
        expected_columns = ['screen_time_hours', 'nighttime_use', 'app_switches', 
                           'social_media_ratio', 'unlocks', 'profile']
        input_data = input_data[expected_columns]
        
        logger.info(f"Reordered columns: {input_data.columns.tolist()}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        logger.info(f"Prediction result: {prediction}")
        
        return {"fatigue_level": prediction}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )
