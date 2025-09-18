import os
import sys
from typing import Optional, Dict, List
from sqlalchemy import select, desc
from mcp.server.fastmcp import FastMCP
from mcp_server.database import db_manager
from dotenv import load_dotenv
from mcp_server.exception import TelcoChurnMCPException


load_dotenv(override=True)

mcp = FastMCP("telco-churn")

@mcp.tool
def get_customer_personal_info(customer_id: str) -> Dict:
    try: 
        table = db_manager.get_table("customers")
        if not table:
            raise TelcoChurnMCPException("customers table not found")
        
        with db_manager.get_pg_connection() as conn:
            query = select(table).where(table.c.customer_id==customer_id)
            result = conn._execute(query).first()
            return dict(result) if result else None 
        
    except Exception as e:
        raise TelcoChurnMCPException(e, sys)
    
@mcp.tool
def get_customer_service_info(customer_id: str) -> Dict:
    try:
        collection = db_manager.get_mongo_collection()
        result = collection.find_one({"customer_id": customer_id})
        if result:
            result["_id"] = str(result["_id"])
        return dict(result) if result else None 
    
    except Exception as e:
        raise TelcoChurnMCPException(e, sys)
    
@mcp.tool
def get_available_offers() -> Dict:
    try:
        table = db_manager.get_table("offers")
        if not table:
            raise TelcoChurnMCPException("Offers table not found")

        with db_manager.get_pg_connection() as conn:
            query = select(table).order_by(desc(table.c.created_at))
            results = conn.execute(query).fetchall()
            return [dict(r) for r in results]
    
    except Exception as e:
        raise TelcoChurnMCPException(e, sys)
    
@mcp.tool
def get_prediction(customer_id: str) -> int:
    try:
        pass 
    except Exception as e:
        raise TelcoChurnMCPException(e, sys)

if __name__ == "__main__":
    mcp.run(transport="stdio")