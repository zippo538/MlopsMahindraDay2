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
    BEDS    : int        
    BATH    : int        
    PROPERTYSQFT : int 
    LOCALITY : str = 'New York'
    
    class Config :
        schema_extra = {
            "example" : {
                "PRICE" : 200000,
                'BEDS' : 2,
                'BATH' : 3,
                'PROPERTYSQFT' : 2000,
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
    with open(Config.ENCODING_PATH, 'rb') as f :
        encoder = pickle.load(f)
    
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model {str(e)}")
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
        #get locality
        
        
        # Prepare input
        feature_dict = features.dict()
        input_df = pd.DataFrame([feature_dict])[Config.FEATURE_COLUMN]
        
        if 'PRICE' in input_df.columns:
            input_df = input_df.drop(columns=['PRICE'])

        
        feature_locality = input_df[['LOCALITY']]
        new_encoded = encoder.transform(feature_locality)
        
        
        # make df encode
        get_name_feature_encode = encoder.get_feature_names_out(['LOCALITY'])
        df_encoded= pd.DataFrame(new_encoded,columns=get_name_feature_encode)
        
        
        #drop_locality
        input_df = input_df.drop(columns='LOCALITY')
        
        #merge encoded
        input_df = pd.concat([input_df.reset_index(drop=True),df_encoded],axis=1)
        
        logger.info(input_df)
        
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