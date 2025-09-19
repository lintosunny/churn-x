import sys
import pandas as pd
from typing import Dict, List
from mcp_server.logger import logging
from mcp_server.exception import TelcoChurnMCPException
from mcp_server.db.mongo_client import get_mongo_collection


def get_customer_business_data(customer_id: str) -> pd.DataFrame:
    """
    Fetch business-related attributes for a given customer from MongoDB.

    Args:
        customer_id (str): The CustomerID to search for.

    Returns:
        Dict: A dictionary containing selected business attributes for the customer.
    """
    try:
        # Get MongoDB collection object
        collection = get_mongo_collection()
        logging.info(f"Fetching business data for customer_id={customer_id}")

        # Include only business-relevant fields
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

        # Fetch single document for the given customer_id
        doc = collection.find_one({"CustomerID": customer_id}, projection)
        if not doc:
            logging.warning(f"No business data found for customer {customer_id}")
            raise ValueError(f"No business data found for customer {customer_id}")
        
        logging.info(f"Successfully fetched business data for customer_id={customer_id}")
        return doc
    
    except Exception as e:
        logging.error(f"Error while fetching business data for customer_id={customer_id}: {e}")
        raise TelcoChurnMCPException(e, sys) from e



if __name__ == '__main__':
    print(get_customer_business_data(customer_id='5940-AHUHD'))