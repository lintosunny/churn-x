import streamlit as st
import requests

st.title("Customer Retention Offer Generator")

# Input field for customer ID
customer_id = st.text_input("Enter Customer ID")

if st.button("Generate Offer"):
    if customer_id:
        try:
            # Call FastAPI backend
            response = requests.post(
                "http://127.0.0.1:8000/generate_offer",
                json={"customer_id": customer_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success("Offer generated successfully!")
                st.subheader("Generated Offer Email:")
                st.text(data["offer_mail"])
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Failed to connect to server: {e}")
    else:
        st.warning("Please enter a Customer ID")
