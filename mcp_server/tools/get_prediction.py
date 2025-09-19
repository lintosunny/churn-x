import os
import sys
import pandas as pd
from mcp_server.db.mongo_client import get_mongo_collection
from mcp_server.constants import MODEL_PATH
from mcp_server.exception import TelcoChurnMCPException
from mcp_server.logger import logging
from mcp_server.utils.main_utils import load_object


def get_prediction(customer_id: str) -> float:
    try:
        collection = get_mongo_collection()

        projection = {
            '_id': 0, 
            'CustomerID': 1, 
            'Senior Citizen': 1, 
            'Partner': 1, 
            'Dependents': 1, 
            'Tenure Months': 1,
            'Phone Service': 1, 
            'Multiple Lines': 1, 
            'Internet Service': 1, 
            'Online Security': 1, 
            'Online Backup': 1,
            'Device Protection': 1, 
            'Tech Support': 1, 
            'Streaming TV': 1, 
            'Streaming Movies': 1, 
            'Contract': 1, 
            'Paperless Billing': 1, 
            'Payment Method': 1, 
            'Monthly Charges': 1, 
            'additional_services': 1,
            '3rd_party_services': 1, 
            'is_payment_automatic': 1, 
            'is_payment_recurring': 1,
            'CLTV': 1,
            'City': 1,
            'Total Charges': 1
        }

        doc = collection.find_one({"CustomerID": customer_id}, projection)
        if not doc:
            logging.warning(f"No business data found for customer {customer_id}")
            raise ValueError(f"No business data found for customer {customer_id}")
        
        logging.info(f"Successfully fetched business data for customer_id={customer_id}")
        x = pd.DataFrame([doc])

        model = load_object(MODEL_PATH)
        prediction = model.predict_proba(x)
        logging.info(f"prediction completed. {customer_id}: {prediction}")

        return prediction
        
    except Exception as e:
        raise TelcoChurnMCPException(e, sys)


    
if __name__ == '__main__':
        print(get_prediction(customer_id='5940-AHUHD'))