import os 
import sys
import json
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from typing import Dict

# Add project root (root/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcp_server.main import (
    get_customer_business_data,
    get_customer_personal_data,
    get_offers,
    get_prediction
)

load_dotenv(override=True)

groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

def generate_offer_mail(customer_id: str) -> Dict:

    email_agent = Agent(
        model=Groq(api_key=groq_api_key),
        description=(
        "An AI-powered sales assistant that crafts personalized offer letters to retain customers. "
        "It leverages customer personal and business data, churn likelihood, and available offers, "
        "applying principles of reciprocity, scarcity, social proof, and emotional appeal to maximize engagement."
        ),
        tools=[get_customer_business_data, get_customer_personal_data, get_offers, get_prediction]
    )

    prompt = f"""
        Generate a personalized offer email for the provided {customer_id} as input using the available tools. 

        Steps:
        1. Fetch personal data with personal_data_tool.
        2. Fetch business data with business_data_tool.
        3. Fetch churn probability with prediction_tool (use it to personalize the subject, do NOT include it in the email body).
        4. Fetch available offers with available_offers_tool.
        5. Select the top 3 relevant offers based on customer profile.
        6. Compose a professional email:
        - Include "Send to:" with the customer's email.
        - Start with "Dear name of the customer id,"
        - Create a compelling, personalized subject line using customer data and churn score.
        - List the selected offers with their descriptions.
        - Include a call-to-action to activate the offers.
        - End with a polite closing.
        - Don't add churn probability anywhere in email body or subject
        - sender is Telco Co.
        
        Return the complete email text ready to send.
    """

    response = email_agent.run(prompt)

    churn_proba = get_prediction(customer_id)

    return {
        "customer_id": customer_id,
        "churn_score": churn_proba,
        "offer_mail": response.content
    }



if __name__ == '__main__':
    x = generate_offer_mail(customer_id='5940-AHUHD')
    print(x)