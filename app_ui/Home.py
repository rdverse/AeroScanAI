import streamlit as st
from PIL import Image

import streamlit as st

# Set page title and icon
st.set_page_config(
    page_title="VivaVerve - Your Health Companion",
    page_icon="💊",
)

# Add a colorful banner with a gradient background
st.markdown(
    """
    <style>
    .banner {
        background-image: linear-gradient(45deg, #FF6B6B, #FFC3A0);
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
        <p class="banner-text">Welcome to VivaVerve</p>
        <p class="sub-banner-text">Your Multimodal Healthcare Companion</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add an animated GIF or illustration here
st.image("assets/healthcare.gif")

# Add an engaging heading
st.header('Unlock a Healthier You with VivaVerve')

# Add animated emojis for extra flair
st.markdown('🌟 Explore AI-powered healthcare solutions 🚀')

# Create colorful card sections with hover animations
col1, col2, col3, col4 = st.columns(4)

# Add icons and hover effects for each feature
with col1:
    #st.image("heart.png", use_container_width=True)
    st.subheader("Personalized Health Insights")
    st.caption('🔍 Get tailored health insights based on your data.')

with col2:
    #st.image("medicine.png", use_container_width=True)
    st.subheader("Medication Management")
    st.caption('💊 Manage your medications efficiently with AI assistance.')

with col3:
    #st.image("fitness.png", use_container_width=True)
    st.subheader("Fitness Tracking")
    st.caption('🏋️‍♂️ Track your fitness goals and progress effortlessly.')

with col4:
    #st.image("chatbot.png", use_container_width=True)
    st.subheader('Healthcare Chatbot')
    st.caption('💬 Ask questions and get instant healthcare advice.')

# Add an interactive button for a demo or sign-up
if st.button("Try a Demo"):
    # Replace with your demo code or link
    st.write("Demo link will be here.")

# Add colorful sections or banners for important notices
st.markdown(
    """
    <div style="background-color:#FF9F43;padding:20px;border-radius:10px;margin-top:30px">
    <h3 style="color:white;text-align:center;">Important Notices & Disclaimers</h3>
    <p style="color:white;text-align:center;">Our AI solutions are designed to assist, not replace, professional healthcare advice. Performance may vary based on individual data. Please consult with a healthcare professional for medical advice.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add more interactive elements or animations as needed
