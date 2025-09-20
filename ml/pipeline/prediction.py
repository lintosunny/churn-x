from ml.data_access.telco_data import TelcoData
from ml.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME
from ml.utils.main_utils import load_object
from ml.utils.ml_utils.estimator import ModelResolver
from ml.logger import logging 
from ml.exception import TelcoChurnMLException
import os 
import sys

model_resolver = ModelResolver()
telco_data = TelcoData()

def predict(customer_id: str) -> float:
    try:
        # Fetch features for the customer
        x = telco_data.get_data_for_prediction(
            collection_name=DATA_INGESTION_COLLECTION_NAME, 
            database_name=DATA_INGESTION_DATABASE_NAME, 
            customer_id=customer_id)
        
        # Get the best model path   
        best_model_path = model_resolver.get_best_model_path()
        if not best_model_path or not os.path.exists(best_model_path):
            raise TelcoChurnMLException(f"No valid model found at path: {best_model_path}")
        
        # Load model
        model = load_object(best_model_path)

        # Predict churn score
        churn_score = model.predict_proba(x)
        
        logging.info(f"Churn prediction for customer {customer_id}: {churn_score}")
        return float(churn_score)
    
    except Exception as e:
        logging.error(f"Error predicting churn for customer {customer_id}: {e}")
        raise TelcoChurnMLException(e, sys)


if __name__ == '__main__':
        print(predict(customer_id='5940-AHUHD'))