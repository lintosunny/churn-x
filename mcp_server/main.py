from mcp.server.fastmcp import FastMCP
from mcp_server.tools.get_customer_personal_data import get_customer_personal_data
from mcp_server.tools.get_customer_business_data import get_customer_business_data
from mcp_server.tools.get_offers import get_offers
from mcp_server.tools.get_prediction import get_prediction
from dotenv import load_dotenv

load_dotenv(override=True)


mcp = FastMCP("telco-churn")

@mcp.tool()
def business_data_tool(customer_id: str):
    """Fetch business-related customer data."""
    return get_customer_business_data(customer_id)

@mcp.tool()
def personal_data_tool(customer_id: str):
    """Fetch personal-related customer data."""
    return get_customer_personal_data(customer_id)

@mcp.tool()
def prediction_tool(customer_id: str):
    """Predict churn probability for a customer."""
    return get_prediction(customer_id)

@mcp.tool()
def available_offers_tool():
    """Get available offers"""
    return get_offers()


if __name__ == '__main__':
    mcp.run(transport='stdio')