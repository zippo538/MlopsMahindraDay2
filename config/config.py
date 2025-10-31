import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    ARTIFACTS_DIR = BASE_DIR / "artifact"
    LOGS_DIR = BASE_DIR / "logs"
    STATIC_DIR = BASE_DIR / "static"
    
    # Data paths
    DATA_PATH = ARTIFACTS_DIR / "NY-House-Dataset.csv"
    MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
    METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
    
    # model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TARGET_COLUMN = "PRICE"
    
    
    
    #drop value
    DROP_VALUE_PRICE = [304, 1, 317, 310, 360, 463]
    DROP_VALUE_BEDS = [1143, 2653, 3276,2488,765,2265,3589,3603,4240,2564,1092,1410,1214,750,4191,4404]
    DROP_VALUE_PROPERTYSQFT = [4623,2146,2148,823,3130,141,2932,2054,917,1823,4353,2107,69,4,2171]
    
    # drop columns
    DROP_COLUMNS = ['TYPE','STATE','ADDRESS','MAIN_ADDRESS','ADMINISTRATIVE_AREA_LEVEL_2','STREET_NAME', 'LONG_NAME', 'FORMATTED_ADDRESS','LONGITUDE', 'LATITUDE','BROKERTITLE','SUBLOCALITY']
    
    # change type int 
    BATH = 'BATH'
    PROPERTYSQFT = 'PROPERTYSQFT'

    
    # Feature columns for model XGBoost 
    PARAMS = {
        'n_estinamtors' : [100,200,300,400],
        'max_depth'  : [2,4,6,8,10],
        'learning_rate' : [0.05, 0.01, 0.1], 
        }
    CV_FOLDS = 5
    SCORING = 'neg_mean_squared_error'
    
    