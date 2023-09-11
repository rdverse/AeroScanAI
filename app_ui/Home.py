import streamlit as st

# Set page title and icon
st.set_page_config(
    page_title="AeroScan.AI - Aerospace Anomaly Detection",
    page_icon="✈️",
)

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

st.header('Empowering Aerospace Industry with Anomaly Detection')

st.markdown('🌟 Explore AI-powered anomaly detection solutions 🚀')

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Real-time Active Learning")
    st.caption('🔍 Monitor aircraft systems for anomalies through active learning or supervised learning based on labelled data availability.')

with col2:
    st.subheader("Waveform Analysis")
    st.caption('🚁 Learn why? is a pixel identified as a defect by training a simple ML model and inspecting the local feature importance')

with col3:
    st.subheader("Classify Defect Types")
    st.caption('📈 Add new quality check parameters and train a model with tabular data for defect type.')

with col4:
    st.subheader('')
    st.caption('🛡️ ')


st.markdown(
    """
    <div style="background-color:#FF9F43;padding:20px;border-radius:10px;margin-top:30px">
    <h3 style="color:white;text-align:center;">Important Notices & Disclaimers</h3>
    <p style="color:white;text-align:center;">AeroScan.AI is designed as an additional tool to enhance aerospace NDI safety. Always follow industry regulations and consult with industrial experts for critical decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)