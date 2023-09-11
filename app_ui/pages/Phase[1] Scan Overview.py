import streamlit as st
from PIL import Image
import requests
import ast
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from skimage.transform import resize

#Set the title of the app
st.title('Scan Anomaly Detection')

# Create two tabs: "Application" and "Help"
app_tab, help_tab = st.tabs(["Application", "Help"])
# "Application" tab
with app_tab:
    # Header image
    col11, col22 = st.columns(2)
    with col11:
        image = Image.open('./assets/fuselage.jpg')
        st.image(image)
        st.markdown("""
                    ###### 3D-UNet model with adaptive input training and active learning.
                    """)
    with col22:
        st.markdown("""
            ##### Phase 1 : Train a model using 3D scans with defects and without defects.
            Here we also employ an active learning approach that relies on the user to set a prediction threshold and learn to differentiate between defects and no defects without labels.
            This gives an overview of the data and the areas that might potentially have a defect.   
            """)
        
    cola1, cola2 = st.columns(2)
    
    with cola1:
        st.info("Model Parameters")
        model_name = st.text_input('Model Name',key='model name', help='The name of the model (change when re-training)', value='model')
        model_path = st.text_input('Model Save Path', key='model path', value='./box/models/scan_anomaly/')
        active_learning = st.selectbox('Active Learning', ["True", "False"], placeholder="False")
        al_threshold = st.slider('Active learning prediction Threshold',min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        #test_scan = st.selectbox('Test Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan", "random"], placeholder="low_defect_scan")
        st.info("Training Parameters")
        n_cpus = st.slider('Number of CPUs',min_value=1, max_value=128, value=1, step=1)
        n_epochs = st.slider('Number of Epochs',min_value=1, max_value=100, value=3, step=1)
        batch_size = st.slider('Batch Size',min_value=1, max_value=128, value=32, step=1)

    with cola2:    
        st.info("Data Parameters")
        n_channels = st.slider('Number of channels in waveform',min_value=1, max_value=124, value=10, step=1)
        n_classes = st.selectbox('Number of classes', [1], placeholder="1")
        n_samples = st.slider('Number of samples',min_value=100, max_value=10000, value=100, step=100)
        img_dim = st.selectbox('Image Dimension', [16,32,64,128], placeholder="64")
        percent_test = st.slider('Percentage of data saved for Testing',min_value=0.1, max_value=0.5, value=0.3, step=0.1)
        #with st.expander("More info on data"):
    
    # Button to train the model
    if st.button('Train Model', key='training'):
        # Build the request
        URL = 'http://scan_anomaly:5003/train'
    
        DATA = {'active_learning':active_learning,
                'al_threshold':al_threshold,  
                'img_dim':img_dim, 
                'n_channels':n_channels,
                'n_classes':n_classes,
                'n_samples':n_samples,
                'model_name':model_name, # 'model
                'model_path':model_path, 
                'n_cpus': n_cpus,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'percent_test': percent_test
                }
        TRAINING_RESPONSE = requests.post(url=URL, json=DATA)
        
        st.divider()
        st.markdown(
            """
            #### Model Training results
            """    )
    
        if len(TRAINING_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
            st.info(TRAINING_RESPONSE.text)
        else:
            st.info(TRAINING_RESPONSE.json().get('msg'))
            st.table(pd.DataFrame.from_dict(TRAINING_RESPONSE.json().get('results')))
        
    st.divider()
    st.markdown(
        """
        #### Visualize the defects on a test scan
        """    )     
    al_threshold = st.slider('Active inference prediction Threshold',min_value=0.1, max_value=0.9, value=0.5, step=0.1)    
    test_scan = st.selectbox('Test Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan", "random"], placeholder="low_defect_scan")
    if st.button('Visualize', key='predict_visualize'):
        URL = 'http://scan_anomaly:5003/predict'
        DATA = {'al_threshold':al_threshold, 
                'model_name': model_name, 
                'model_path':model_path,
                'n_channels': n_channels, 
                'img_dim': img_dim, 
                'test_scan' : test_scan
                }
        INFERENCE_RESPONSE = requests.post(url = URL, json = DATA)
        #print(INFERENCE_RESPONSE.json())
        #st.info(INFERENCE_RESPONSE.text)
        ####################uncomment
        #Check the response status code
        if len(INFERENCE_RESPONSE.text) < 40:       
            st.error("Model Inference Failed")
            st.info(INFERENCE_RESPONSE.text)
        else:
            st.success('Inference was Succesful')
            #st.info(INFERENCE_RESPONSE.text)
            preds = np.array(ast.literal_eval(INFERENCE_RESPONSE.json().get('preds')))*255
            labels = np.array(ast.literal_eval(INFERENCE_RESPONSE.json().get('labels'))).squeeze()*255
            inputs = np.array(ast.literal_eval(INFERENCE_RESPONSE.json().get('inputs'))).squeeze()
            colb1, colb2, colb3 = st.columns(3)
            preds, labels, inputs = resize(preds, (64,64), anti_aliasing=True), resize(labels, (64,64), anti_aliasing=True), resize(inputs, (64,64), anti_aliasing=True)
    #        with colb1:
            st.info("Input test scan : {} ".format(test_scan))
            #st.image(inputs)
        # Create a Plotly heatmap for image display
            figI = px.imshow(inputs, color_continuous_scale="viridis")
            figI.update_layout(
                title="Test scan under analysis : {} ".format(test_scan),
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
            )
            st.plotly_chart(figI)
    
    #        with colb2:
            st.info("Model prediction")
            figP = px.imshow(preds, color_continuous_scale="gray")
            figP.update_layout(
                title="Predicted defects (in white pixels)",
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
            )
            st.plotly_chart(figP)
    #        st.image(preds)
    
    #        with colb3:
            #st.info("Ground Truth")
            figL = px.imshow(labels, color_continuous_scale="gray")
            figL.update_layout(
                title="Ground truth the scan (white pixels are defects))",
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
            )
            st.plotly_chart(figL)
            
            with st.expander('Note'):
                st.info(""" 1. Due to the screen size limitations all images are scaled to 64px for consistency.
                            2. The input test scan that has a 3d structure is converted to a 2d image by computing the standard deviation for plotting.
                        """)

with help_tab:
    st.markdown("#### Importance of Waveform Probing")
    st.markdown(        """
        #### This is the start of inspection where a model predicts on a scan-level if there is a defect or not.
        ##### The model takes input as a 3D scan. 
        ##### Each pixel is a waveform of n points defined by the user.
        ##### Thanks to UNet's adaptable architecture, by adjusting the layers according to the input size, we can use the same model architecture for different input sizes.
        ##### However, it is important to note that the smaller the input, smaller will be the model. 
    """)