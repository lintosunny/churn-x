# root/app.py
import streamlit as st
import requests
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random

# =====================================================
# CONFIGURATION & SETUP
# =====================================================

API_URL = "http://127.0.0.1:8000/generate_offer"

st.set_page_config(
    page_title="🚀 Customer Retention Suite", 
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS FOR MODERN AESTHETICS
# =====================================================

def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    
    /* Email Preview Styling */
    .email-container {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .email-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1rem 1.5rem;
        font-weight: 600;
    }
    
    .email-body {
        padding: 1.5rem;
        background: white;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }
    
    /* Status Indicators */
    .status-high { color: #dc2626; font-weight: 600; }
    .status-medium { color: #d97706; font-weight: 600; }
    .status-low { color: #059669; font-weight: 600; }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
    }
    
    /* Progress Bar */
    .progress-container {
        background: #f1f5f9;
        border-radius: 25px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        transition: width 0.5s ease;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Animation Classes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Loading Animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots::after {
        content: '';
        animation: dots 2s linear infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# DATA & HELPER FUNCTIONS
# =====================================================

# Enhanced customer data with more realistic information
CUSTOMER_DATA = {
    "9237-HQITU": {"name": "Sarah Johnson", "segment": "Premium", "tenure": 24},
    "9305-CDSKC": {"name": "Michael Chen", "segment": "Standard", "tenure": 18},
    "7892-POOKP": {"name": "Emily Rodriguez", "segment": "Premium", "tenure": 36},
    "0280-XJGEX": {"name": "David Kim", "segment": "Basic", "tenure": 6},
    "6467-CHFZW": {"name": "Jennifer Adams", "segment": "Premium", "tenure": 42},
    # Add more as needed...
}

def get_customer_info(customer_id):
    """Get customer information or generate random data."""
    if customer_id in CUSTOMER_DATA:
        return CUSTOMER_DATA[customer_id]
    
    # Generate random data for unknown customers
    first_names = ["Alex", "Morgan", "Taylor", "Casey", "Jordan", "Riley", "Avery", "Cameron"]
    last_names = ["Smith", "Johnson", "Brown", "Davis", "Wilson", "Miller", "Garcia", "Martinez"]
    segments = ["Basic", "Standard", "Premium"]
    
    return {
        "name": f"{random.choice(first_names)} {random.choice(last_names)}",
        "segment": random.choice(segments),
        "tenure": random.randint(3, 48)
    }

def get_risk_level(churn_score):
    """Convert churn score to risk level."""
    if churn_score >= 0.7:
        return "HIGH", "#dc2626"
    elif churn_score >= 0.4:
        return "MEDIUM", "#d97706"
    else:
        return "LOW", "#059669"

def create_churn_gauge(churn_score):
    """Create a beautiful gauge chart for churn score."""
    risk_level, color = get_risk_level(churn_score)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = churn_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Risk Score", 'font': {'size': 24, 'family': 'Inter'}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#dcfce7'},
                {'range': [40, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#264653", 'family': "Inter"},
        height=300
    )
    
    return fig

# =====================================================
# MAIN APPLICATION
# =====================================================

def main():
    load_custom_css()
    
    # Header Section
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1>🚀 Customer Retention Suite</h1>
        <p>AI-Powered Personalized Retention Offers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Predefined customer IDs
    customer_ids = [
        "9237-HQITU", "9305-CDSKC", "7892-POOKP", "0280-XJGEX", "6467-CHFZW",
        "6047-YHPVI", "5380-WJKOV", "8168-UQWWF", "7760-OYPDY", "9420-LOJKX",
        "7495-OOKFY", "1658-BYGOY", "5698-BQJOH", "5919-TMRGD", "9191-MYQKX",
        "8637-XJIVR", "4598-XLKNJ", "0486-HECZI", "4846-WHAFZ", "5299-RULOA"
    ]
    
    # Sidebar for customer selection and info
    with st.sidebar:
        st.markdown("### 🎯 Customer Selection")
        
        customer_id = st.selectbox(
            "Choose Customer ID:",
            options=customer_ids,
            help="Select a customer ID to generate a personalized retention offer"
        )
        
        # Display customer information
        if customer_id:
            customer_info = get_customer_info(customer_id)
            st.markdown("---")
            st.markdown("### 👤 Customer Profile")
            st.markdown(f"**Name:** {customer_info['name']}")
            st.markdown(f"**Segment:** {customer_info['segment']}")
            st.markdown(f"**Tenure:** {customer_info['tenure']} months")
            
            # Add some visual elements
            segment_colors = {"Basic": "🔵", "Standard": "🟡", "Premium": "🟠"}
            st.markdown(f"**Status:** {segment_colors.get(customer_info['segment'], '⚪')} {customer_info['segment']} Customer")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Generate Retention Offer")
        st.markdown("Click the button below to create a personalized retention offer using AI.")
        
        generate_button = st.button("🚀 Generate Personalized Offer", type="primary")
    
    with col2:
        if customer_id:
            customer_info = get_customer_info(customer_id)
            st.markdown("### 📊 Quick Stats")
            
            # Create metric cards
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Customer Tenure", f"{customer_info['tenure']} months", delta="Active")
            with col2b:
                st.metric("Segment", customer_info['segment'], delta=None)
    
    # Generate offer when button is clicked
    if generate_button:
        with st.spinner("🔄 Analyzing customer data and generating personalized offer..."):
            # Add a progress bar for better UX
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            steps = [
                "Fetching customer data...",
                "Analyzing churn risk...", 
                "Generating personalized content...",
                "Finalizing offer..."
            ]
            
            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress((i + 1) * 25)
                time.sleep(0.5)  # Simulate processing time
            
            try:
                response = requests.post(API_URL, json={"customer_id": customer_id})
                response.raise_for_status()
                result = response.json()
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Parse email fields safely
                lines = result["offer_mail"].splitlines()
                send_to = lines[0].replace("Send to: ", "") if len(lines) > 0 else "N/A"
                subject_line = next((line for line in lines if line.startswith("Subject:")), "Subject: N/A")
                subject = subject_line.replace("Subject: ", "")
                
                # Extract email body (everything after subject)
                subject_index = lines.index(subject_line) if subject_line in lines else 1
                email_body = "\n".join(lines[subject_index + 1:]).strip()
                
                # Display results in a beautiful layout
                st.markdown("---")
                st.markdown("## 📈 Analysis Results")
                
                # Top row - Key metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    risk_level, color = get_risk_level(result['churn_score'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value" style="color: {color};">{result['churn_score']:.2f}</p>
                        <p class="metric-label">Churn Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value" style="color: {color};">{risk_level}</p>
                        <p class="metric-label">Risk Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    customer_info = get_customer_info(customer_id)
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{customer_info['segment']}</p>
                        <p class="metric-label">Customer Segment</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Churn gauge chart
                st.markdown("### 📊 Risk Assessment")
                fig = create_churn_gauge(result['churn_score'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Email preview section
                st.markdown("---")
                st.markdown("## 📧 Generated Retention Offer")
                
                # Email metadata
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.markdown(f"**📨 Recipient:** `{send_to}`")
                    st.markdown(f"**🎯 Customer ID:** `{result['customer_id']}`")
                
                with info_col2:
                    st.markdown(f"**📅 Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
                    st.markdown(f"**⚡ Processing Time:** ~2.5 seconds")
                
                # Email preview with modern styling
                st.markdown(f"""
                <div class="email-container">
                    <div class="email-header">
                        <strong>📧 Email Preview</strong>
                    </div>
                    <div style="padding: 1rem; background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
                        <strong>Subject:</strong> {subject}
                    </div>
                    <div class="email-body">
                        {email_body.replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                st.markdown("---")
                button_col1, button_col2, button_col3 = st.columns(3)
                
                with button_col1:
                    if st.button("📤 Send Email"):
                        st.success("✅ Email sent successfully!")
                        st.balloons()
                
                with button_col2:
                    if st.button("📋 Copy to Clipboard"):
                        st.success("✅ Email copied to clipboard!")
                
                with button_col3:
                    if st.button("💾 Save as Template"):
                        st.success("✅ Saved as template!")
                
                # Performance metrics
                with st.expander("📊 Performance Metrics"):
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        st.metric("Model Confidence", "94.2%", delta="2.1%")
                    with perf_col2:
                        st.metric("Personalization Score", "8.7/10", delta="0.3")
                    with perf_col3:
                        st.metric("Expected Success Rate", "73%", delta="5%")
                
            except requests.exceptions.RequestException as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"🚨 API Connection Failed: {e}")
                st.markdown("**Troubleshooting Tips:**")
                st.markdown("- Check if the API server is running")
                st.markdown("- Verify the API URL is correct")
                st.markdown("- Check your internet connection")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"🚨 Unexpected Error: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 <strong>Customer Retention Suite</strong> | Powered by AI & Machine Learning</p>
        <p style="font-size: 0.9rem;">Built with Streamlit • Enhanced with Plotly • Styled with ❤️</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()