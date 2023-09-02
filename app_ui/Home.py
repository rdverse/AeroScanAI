import streamlit as st

# Set page title and icon
st.set_page_config(
    page_title="AeroScan.AI - Aerospace Anomaly Detection",
    page_icon="✈️",
)

# Add a colorful banner with a gradient background
st.markdown(
    """
    <style>
    .banner {
        background-image: linear-gradient(45deg, #3399FF, #66CCFF);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .banner-text {
        color: white;
        font-size: 36px;
        font-weight: bold;
    }
    .sub-banner-text {
        color: white;
        font-size: 24px;
    }
    </style>
    <div class="banner">
        <p class="banner-text">Welcome to AeroScan.AI</p>
        <p class="sub-banner-text">Your Aerospace Anomaly Detection Solution</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add an engaging heading
st.header('Empowering Aerospace Industry with Anomaly Detection')

# Add animated emojis for extra flair
st.markdown('🌟 Explore AI-powered anomaly detection solutions 🚀')

# Create colorful card sections with hover animations
col1, col2, col3, col4 = st.columns(4)

# Add icons and hover effects for each feature
with col1:
    st.subheader("Real-time Monitoring")
    st.caption('🔍 Monitor aircraft systems for anomalies in real-time.')

with col2:
    st.subheader("Maintenance Alerts")
    st.caption('🚁 Receive instant alerts for maintenance needs.')

with col3:
    st.subheader("Data Analytics")
    st.caption('📈 Analyze aircraft data for performance insights.')

with col4:
    st.subheader('Safety Assurance')
    st.caption('🛡️ Ensure safety with advanced anomaly detection.')

# Add an interactive button for a demo or sign-up
if st.button("Try a Demo"):
    # Replace with your demo code or link
    st.write("Demo link will be here.")

# Add colorful sections or banners for important notices
st.markdown(
    """
    <div style="background-color:#FF9F43;padding:20px;border-radius:10px;margin-top:30px">
    <h3 style="color:white;text-align:center;">Important Notices & Disclaimers</h3>
    <p style="color:white;text-align:center;">Our AI solutions are designed to enhance aerospace safety. Always follow industry regulations and consult with aviation experts for critical decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add more interactive elements or animations as needed
