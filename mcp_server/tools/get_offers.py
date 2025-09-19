import sys
from mcp_server.db.postgres_client import get_postgres_connection
from mcp_server.logger import logging
from typing import Dict, List
from mcp_server.exception import TelcoChurnMCPException


def get_offers() -> List[Dict]:
    """
    Fetch all available offers from the PostgreSQL database

    Returns:
        List[Dict]: A list of offers, where each offer is represented as a dictionary
    """
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()

        # SQL query to fetch all offers
        query = "SELECT offer_id, offer_name, offer_description, categories FROM offers"
        logging.info(f"Executing query: {query}")
        cursor.execute(query)

        rows = cursor.fetchall()
        logging.info(f"Fetched {len(rows)} offers from database")

        conn.close()
        logging.info("Postgres connection closed")

        # Convert rows into list of dicts
        offers = []
        for r in rows:
            offers.append({
                "offer_id": r[0],
                "offer_name": r[1],
                "offer_description": r[2],
                "categories": r[3]
            }) 

        return offers   
    
    except Exception as e:
        raise TelcoChurnMCPException(e, sys)



if __name__ == '__main__':
    print(get_offers())