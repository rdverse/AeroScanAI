import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
import json
# Set the title of the app
st.title('Waveform Probing')

# Create two tabs: "Application" and "Help"
app_tab, help_tab = st.tabs(["Application", "Help"])

# "Application" tab
with app_tab:
    # Header image
    col11, col22 = st.columns(2)
    with col11:
        image = Image.open('./assets/waveform2.png')
        st.image(image)
        st.markdown("""
                    ###### Phase 2 : Waveform model training and inference
                    """)
        
    with col22:
        st.markdown("""
            ##### Phase 2 : Analyze waveform of each pixel in a 3D scan. 
            Specifically, train a random forest model on with raw waveform data and 
            use explainable-AI for inference and learn what parts of waveform were given most importance.   
            """)
        
    cola1, cola2 = st.columns(2)
    with cola1:
        test_scan = st.selectbox('Test Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan", "random"], placeholder="low_defect_scan")
        train_scan = st.selectbox('Train Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan", "random"], placeholder="high_defect_scan")
        model_path = st.text_input('Model Save Path', key='model path', value='./box/models/waveform_probe/')
        
    with cola2:    
        n_channels = st.slider('Number of channels in waveform',min_value=1, max_value=128, value=10, step=1)
        img_dim = st.selectbox('Image Dimension', [8,16,32,64,128], placeholder="64")
        model_name = st.text_input('Model Name',key='model name', help='The name of the model (change when re-training)', value='model')
        with st.expander("More info on data"):
            st.info("""
                   ###### Recomended : Choose test_scan and train_scan with low, medium and high defect scans.
                   ######              These scans are persisted through the three phases, so you can choose the same scans for all phases.
                   ######             (It is advised to choose high_defect_scan for training and others for testing (due to data imbalance in low/medium defect scans)")
                   ###### Choose the following data file if you loaded csv prior to running docker ./box/datasets/defect_classify/train.csv
                    """)
    # Separator
    st.divider()
    st.markdown(
        """
        #### Model training results
        """    )
    # Button to train the model
    if st.button('Train Model', key='training'):
        # Build the request
        URL = 'http://waveform_probe:5002/train'
    
        DATA = {  'img_dim':img_dim, 
                'n_channels':n_channels,
                'test_scan':test_scan,
                'train_scan':train_scan,
                'model_name':model_name, # 'model
                'model_path':model_path, 
                'append_path' : "", # can be discarded - appending data at in the next phase makes more sense
                'ncpu': 1}
        print(DATA)
        TRAINING_RESPONSE = requests.post(url=URL, json=DATA)
        # Check the response status code
        if len(TRAINING_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
        else:
            st.success(TRAINING_RESPONSE.json().get('msg'))
            bench_dict = TRAINING_RESPONSE.json().get('benchmarks')
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