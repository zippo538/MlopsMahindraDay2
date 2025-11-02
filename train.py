from src.data_preparation import load_prepare_data
from src.model import create_pipeline,train_model
from src.evaluation import evaluate_model
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger('train')

def main() :
    try :
        #load and prepare data
        logger.info("Loading and Preparing Data...")
        X_train, X_test, y_train, y_test = load_prepare_data()
        
        #create an train model
        logger.info("Creating and Traning Model...")
        pipeline = create_pipeline()
        model = train_model(pipeline,X_train,y_train)
        
        # evaluate model
        logger.info("Evaluasi model...")
        metrics = evaluate_model(model,X_test,y_test)
        
        logger.info("Training completed successfully")
        logger.info(f"Test MAE: {metrics['MAE']:.4f}")
        logger.info(f"Test RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"Test R2 Score: {metrics['R2_SCORE']:.4f}")
        
    except Exception as e:
        logger.error(f"Error Train Data {e}")
        raise
if __name__ == '__main__':
    main()