from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from config.config import Config
from utils.logger import setup_logger
import pickle

logger = setup_logger('model')

def create_pipeline():
    
    return Pipeline([
        ('regressor',XGBRegressor(
            random_state=Config.RANDOM_STATE,
            n_estimators=200,
            learning_rate=0.1,
            n_jobs=-1
        ))
    ]) 
    
def train_model(pipeline, X_train, y_train):
    try :
        logger.info("Starting model training with GridSearchCV")
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=Config.PARAMS,
            cv=Config.CV_FOLDS,
            scoring=Config.SCORING
        )
        
        grid_search.fit(X_train,y_train)
        
        logger.info(f"Best parameters : {grid_search.best_params_}")
        logger.info(f"Best Score : {grid_search.best_score_}")
        
        #get feature importance from the best model
        
        best_model = grid_search.best_estimator_
        
        with open(Config.MODEL_PATH, 'wb') as f:
            pickle.dump(best_model,f)        
        
        return best_model
         
        
        
    except Exception as e:
        logger.error(f"Error Train Model {e}")
        raise