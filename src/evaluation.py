from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('evaluation')

def evaluate_model(model,X_test,y_test):
    try:
        y_pred = model.predict(X_test)
        
        metrics = {
            'MAE' : mean_absolute_error(y_test,y_pred),
            'RMSE' : mean_squared_error(y_test,y_pred),
            'R2_SCORE': r2_score(y_test,y_pred)
        }
        #save metrics
        with open(Config.METRICS_PATH, 'w') as f:
            json.dump(metrics,f,indent=4)
        logger.info("Model evaluation completed and saved")
        return metrics
        
    except Exception as e:
        logger.error(f"Error Evaluate model {e}")
        raise