from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('api')

class FeatureInput(BaseModel):
    PRICE   : int       
    BEDS    :  int        
    BATH    : int        
    PROPERTYSQF : int 
    LOCALITY : str = 'New York'
    
    class Config :
        schema_extra = {
            "example" : {
                "PRICE" : 200000,
                'BEDS' : 2,
                'BATH' : 3,
                'PROPERTYSQF' : 2000,
                'LOCALITY' : 'New York'
                
            }
        }
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)

#load model 
try:
    with open(Config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

@app.post("/predict")
async def predict(features: FeatureInput):
    try:
        # Validate input
        for feature, value in features.dict().items():
            if not Config.is_valid_feature_value(feature, value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {feature}"
                )
        
        # Prepare input
        feature_dict = features.dict()
        input_df = pd.DataFrame([feature_dict])[Config.FEATURE_COLUMN]
        
        
        
        # Make prediction
        prediction = model.predict(input_df)
        final_prediction = float(np.exp(prediction[0]))
        
        logger.info(f"Prediction made for input: {feature_dict}")
        return {"prediction": final_prediction}
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)