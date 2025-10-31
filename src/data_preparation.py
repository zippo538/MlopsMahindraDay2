import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('data_preparation')

def load_prepare_data():
    try : 
        logger.info("Loading data from csv")
        df = pd.read_csv(Config.DATA_PATH)
        
        
        #delete value outlier
        df.drop(Config.DROP_VALUE_PRICE, inplace=True)  # PRICE outliers
        df.drop(Config.DROP_VALUE_BEDS, inplace=True)  # BEDS outliers
        df.drop(Config.DROP_VALUE_PROPERTYSQFT, inplace=True)  # PROPERTYSQFT outliers
        
        logger.info("Delete Outlier Value...")
        
        #drop columns 
        df.drop(Config.DROP_COLUMNS, axis=1, inplace=True)
        
        logger.info(f"Delete Columns {Config.DROP_COLUMNS}")
        
        
        # drop duplicate 
        df.drop_duplicates(inplace=True)
        
        logger.info(f"Drop Duplicate Value")
        
        
        #change type int
        df[Config.BATH] = df[Config.BATH].astype(int)
        df[Config.PROPERTYSQFT] = df[Config.PROPERTYSQFT].astype(int)
        
        logger.info(f"Change type int columns BATH and PROPERTYSQFT ")
        
       
        X_train, X_test, y_train, y_test = dataset_preparation(df)
        logger.info(f"Data Preparations and Feature Engineering Successfully")
        
        
        return X_train, X_test, y_train, y_test
        
        
    except Exception as e:
        logger.error(f"Error Feature Engineering : {e}")
        raise

def dataset_preparation(df:pd.DataFrame) :
    try : 
        X = df.drop(Config.TARGET_COLUMN,axis=1)
        y = np.log(df[Config.TARGET_COLUMN])
        
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=Config.TEST_SIZE,random_state=Config.RANDOM_STATE)
        
        #encoding data
        X_train,X_test = encoding_feature(X,X_train,X_test)
        logger.info("Success Encoding Data")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error Dataset Preparations {e}")
        raise
    

def encoding_feature(X:pd.DataFrame,X_train :pd.DataFrame,X_test:pd.DataFrame) :
    try : 
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = X.select_dtypes(include=['int64','float64']).columns.tolist()
        
        logger.info("Seperate Data Numerical and Categorical")
        
        
        one_hot = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        one_hot.fit(X_train[categorical_columns])

        X_train_cat = one_hot.transform(X_train[categorical_columns])
        X_test_cat = one_hot.transform(X_test[categorical_columns])
        
        logger.info("Apply One Hot Encoding")
        

        encoded_categorical_columns = one_hot.get_feature_names_out(categorical_columns)

        X_train_cat_df = pd.DataFrame(X_train_cat, columns=encoded_categorical_columns, index=X_train.index)
        X_test_cat_df = pd.DataFrame(X_test_cat, columns=encoded_categorical_columns, index=X_test.index)

        X_train_num_df = X_train[numerical_columns].reset_index(drop=True)
        X_test_num_df = X_test[numerical_columns].reset_index(drop=True)

        X_train_final = pd.concat([X_train_num_df, X_train_cat_df.reset_index(drop=True)],axis=1)
        X_test_final = pd.concat([X_test_num_df, X_test_cat_df.reset_index(drop=True)],axis=1)
        
        logger.info("Merge Data Encoding to DataFrame")
        
        
        return X_train_final,X_test_final
    except Exception as e:
        logger.error(f"Error Encoding Feature {e}")
        raise
    

    
    
     