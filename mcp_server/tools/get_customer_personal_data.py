import sys
from mcp_server.db.postgres_client import get_postgres_connection
from mcp_server.logger import logging
from mcp_server.exception import TelcoChurnMCPException


def get_customer_personal_data(customer_id: str) -> dict:
    """
    Fetch customer personal data from PostgreSQL.
    
    Args:
        customer_id (str): Customer unique ID.
    
    Returns:
        dict: Customer personal information.
    """
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE customer_id = %s", (customer_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError(f"No customer found with id {customer_id}")
        
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "phone": row[3]
        }
    
    except Exception as e:
        raise TelcoChurnMCPException(e, sys)


if __name__ == '__main__':
    print(get_customer_personal_data(customer_id='5940-AHUHD'))