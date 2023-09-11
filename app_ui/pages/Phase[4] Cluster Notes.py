import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
import json
# Set the title of the app
st.title('Cluster Notes')

# Create two tabs: "Application" and "Help"
app_tab, help_tab = st.tabs(["Application", "Help"])

# "Application" tab
with app_tab:
    # Header image
 #   col11, col22 = st.columns(2)
#    with col11:
    st.markdown("""
                    ###### Phase 4 : Cluster and visualize the inspection notes for different types of scans 
                    """)

    col1, col2 = st.columns(2)
    with col1:
        data_name = st.text_input('Data File Name',key='model name', help='The name of the model (change when re-training)', value='cluster')
    with col2:
        data_path = st.text_input('Data File Path',key='model path', help='The path to the model (change when re-training)', value='./box/datasets/waveform_probe/')

    # Separator  
    st.info("Click Below to Start Training Unsupervised Model")   
    # Button to train the model
    if st.button('Train Model', key='clustering'):
        # Build the request
        URL = 'http://waveform_probe:5004/cluster'
    
        DATA = {'model_name': data_name,
        'model_path':data_path}
        
        print(DATA)
        CLUSTERING_RESPONSE = requests.post(url=URL, json=DATA)
        # Check the response status code
        if len(CLUSTERING_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
        else:
            st.success(CLUSTERING_RESPONSE.json().get('msg'))
            bench_dict = CLUSTERING_RESPONSE.json().get('benchmarks')
            for k, v in bench_dict.items():
                st.info("{}: {:.3f}".format(k, v))
            
    # Separator
    st.divider()
    st.markdown(
        """
        #### Waveform defect prediction and interpretation using explainable-AI.
        """
    )

    model_name = st.text_input('Selected Model Name',key='model name selection', value='model')
    model_path = st.text_input('Selected Model Path',key='model path selection', value='./box/models/waveform_probe/')
    col21, col22, col23 = st.columns(3)

    with col21:
        st.info("Inspector's Input")
        x_coord = st.number_input("x coordinate", min_value=0, max_value=128, step=1, value=10)
    with col22:
        st.info("Enter Y coordinate")
        y_coord = st.number_input('y coordinate', min_value=0, max_value=128, step=1, value=10)
    sample = [{'x_coord':x_coord, 'y_coord':y_coord}]
    
    if st.button('Analyze waveform', key='analysis'):
        URL = 'http://waveform_probe:5002/predict'
        DATA = {'model_name': model_name, 
                'model_path':model_path,
                'n_channels': n_channels, 
                'img_dim': img_dim, 
                'test_scan' : test_scan, 
                'x_coord':x_coord , 
                'y_coord' : y_coord ,
                'num_class':2}
        INFERENCE_RESPONSE = requests.post(url = URL, json = DATA)
        st.success(INFERENCE_RESPONSE.json().get('msg'))
        ####################uncomment
        import ast
        data = ast.literal_eval(INFERENCE_RESPONSE.json().get('wavetoplot'))
        feature_importance = ast.literal_eval(INFERENCE_RESPONSE.json().get('feature_importance'))
        data, feature_importance, time = np.array(data).reshape(-1,1), np.array(feature_importance).reshape(-1,1), np.arange(len(data)).reshape(-1,1)
        dataImpDF = pd.DataFrame(np.hstack((data, feature_importance, time)), columns = ['Amplitude', 'Feature Importance', 'Time'])
        
        st.line_chart(dataImpDF, 
                      y = 'Amplitude',
                      x = 'Time')
        st.bar_chart(dataImpDF, 
                     y = 'Feature Importance',
                     x = 'Time')
        with st.expander('More info on feature importance'):
            st.info("""
                    ###### Feature Importance
                    Here we use LIME (local interpretable model-agnostic model explanation) to explain the model's prediction for a given input (i.e., the raw waveform).
                    """)
    
with help_tab:
    st.markdown("#### Importance of Waveform Probing")
    st.markdown(        """
        #### Why is it important to analyze the waveform of each pixel in a 3D scan?
        #####  The waveform of each pixel in a 3D scan has immense information about the defect or a normal signal. 
        ##### Therefore, we first train an random forest model and pick the optimal configuration using grid search.
        ##### Next, we use explainable-AI such as LIME to interpret the model's prediction for a given input (i.e., the raw waveform).
        ##### This helps us understand what parts of the waveform were given most importance for the prediction.
        ##### To interpret this plot, .
    """)