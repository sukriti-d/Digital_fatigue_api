import joblib
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'digital_fatigue_model.joblib')
model = joblib.load(model_path)

def predict_fatigue_level(user_profile: str, screen_time_hours: float, nighttime_use: int, 
                         app_switches: int, social_media_ratio: float, unlocks: int) -> str:
    """
    Predict fatigue level based on user input parameters.
    
    Args:
        user_profile: User profile type
        screen_time_hours: Total screen time in hours
        nighttime_use: Number of times device was used at night
        app_switches: Number of app switches
        social_media_ratio: Ratio of social media usage
        unlocks: Number of device unlocks
        
    Returns:
        str: Predicted fatigue level
    """
    # Prepare input features
    features = [[
        screen_time_hours,
        nighttime_use,
        app_switches,
        social_media_ratio,
        unlocks
    ]]
    
    # Make prediction
    prediction = model.predict(features)[0]
    return prediction 