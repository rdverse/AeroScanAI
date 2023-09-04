import streamlit as st
from PIL import Image
import requests

# Set the title of the app
st.title('Defect class identification and categorization')

# Create two tabs: "Application" and "Help"
app_tab, help_tab = st.tabs(["Application", "Help"])

with app_tab:
    # Header image
    col11, col22 = st.columns(2)
    with col11:
        image = Image.open('./assets/defect-classification.png')
        st.image(image)
        st.markdown("""
                    ### This module uses XGBoost, daal4py, modin, intel extension for sklearn.
                    Visit help tab for more information on the features and model.
                    """)
    with col22:
        st.markdown(
            """
            ##### Phase 3: The inspectors can train a model on statistical features generated from ultrasonic waveforms in NDT testing data.
            ##### The model can then be used to predict the defect class of new ultrasonic waveforms.
            ##### Furthermore, the inspectors can append new data to the training dataset to interpret model's performance.
            """
        )
    # Separator
    st.divider()

    # Patient readmission prediction model training

    st.markdown(
        """
        #### Aero defect classification model training
        """
    )
    # Input data
    #with st.expander("Input Data"): 
    cola1, cola2 = st.columns(2)
    with cola1:
        test_scan = st.selectbox('Test Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan"], placeholder="low_defect_scan")
        train_scan = st.selectbox('Train Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan"], placeholder="high_defect_scan")
        data_file = st.text_input('Training Data File Path', key='data', value='') 
        model_path = st.text_input('Model Save Path', key='model path', value='./box/models/defect_classify/')
        
        
    with cola2:    
        n_channels = st.slider('Number of channels in waveform',min_value=1, max_value=512, value=10, step=1)
        img_dim = st.selectbox('Image Dimension', [8,16,32,64,128,256,512], placeholder="64")
        model_name = st.text_input('Model Name',key='model name', help='The name of the model (change when re-training)', value='model')
        with st.expander("More info on data"):
            st.info("""
                   ###### Recomended : Choose test_scan and train_scan with low, medium and high defect scans - you don't need to choose data file.
                   ######              These scans are persisted through the three phases, so you can choose the same scans for all phases.
                   ######             (It is advised to choose high_defect_scan for training and others for testing (due to data imbalance in low/medium defect scans)")
                   ###### Choose the following data file if you loaded csv prior to running docker ./box/datasets/defect_classify/train.csv
                    """)
        
    # Button to train the model
    if st.button('Train Model', key='training'):
        # Build the request
        URL = 'http://defect_classify:5001/train'
    
        DATA = {'file':data_file, 
                'model_name':model_name, 
                'model_path':model_path, 
                'test_scan':test_scan,
                'train_scan':train_scan,
                'n_channels':n_channels,
                'img_dim':img_dim, 
                'ncpu': 1} # 
        
        print(DATA)
        TRAINING_RESPONSE = requests.post(url=URL, json=DATA)
        print(TRAINING_RESPONSE)
        st.info(TRAINING_RESPONSE)
        # Check the response status code
        if len(TRAINING_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
            st.info(TRAINING_RESPONSE.text)
        else:
            st.success('Training was Succesful')
            st.info(TRAINING_RESPONSE.text)
            st.info('Model Validation Accuracy Score: ' + str(TRAINING_RESPONSE.json().get('validation scores')))

    # Separator
    st.divider()

    # Patient readmission prediction analysis

    st.markdown(
        """
        #### Aero defect classification analysis
        """
    )

    st.markdown('#### Predictive Maintenance Analysis')
    
    selected_model_path = st.text_input('Selected Model Path',key='model path selection', value='./box/models/defect_classify/model.joblib')

    col21, col22, col23 = st.columns(3)

    
    with col21:
        st.info("Inspector's Input")
        backwall = st.selectbox('Backwall', [0,1])
        ramp = st.selectbox('Ramp', [0,1])
        frontwall = st.selectbox('Frontwall', [0,1])
        geometry = st.selectbox('Geometry', [0,1])
    with col22:
        st.info("waveform characteristics")
        no_peaks = st.number_input('Number of Peaks', min_value=0, max_value=10, step=1)
        noise = st.number_input('Noise', min_value=0.00, max_value=10.00, step=.05, value=5.00)
        max = st.number_input('Maximum Value', min_value=0.00, max_value=10.00, step=.05, value=5.00)
        min = st.number_input('Minimum Value', min_value=0.00, max_value=10.00, step=.05, value=5.00)
        signal_noise_ratio = st.number_input('Signal-to-Noise Ratio', min_value=0.00, max_value=10.00, step=.05, value=5.00)
    #sample = [{'backwall':1, 'frontwall':1, 'ramp':1, 'geometry':1, 'no_peaks':1, 'noise':1, 'max':1, 'min':1, 'signal_noise_ratio':1}]
    
    sample = [{'backwall':backwall, 'frontwall':frontwall, 'ramp':ramp, 'geometry':geometry, 'no_peaks':no_peaks, 'noise':noise, 'max':max, 'min':min, 'signal_noise_ratio':signal_noise_ratio}]

    if st.button('Run Maintenance Analysis', key='analysis'):
        URL = 'http://defect_classify:5001/predict'
        DATA = {'data':sample, 'model_path':model_path, 'num_class':3}
        INFERENCE_RESPONSE = requests.post(url = URL, json = DATA)
        st.info("another trail")
        st.info(INFERENCE_RESPONSE)
        st.info(INFERENCE_RESPONSE.text)
        if len(INFERENCE_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
            st.info(INFERENCE_RESPONSE.text)
        else:
            st.success(str(INFERENCE_RESPONSE.json().get('Maintenance Recommendation')))
    
    data_path ="./box/datasets/defect_classify/train.csv"
    #For appending data    
    target = st.selectbox('Select defect category before append', [0,1,2])
    sample_append = [{'backwall':backwall, 'frontwall':frontwall, 'ramp':ramp, 'geometry':geometry, 'no_peaks':no_peaks, 'noise':noise, 'max':max, 'min':min, 'signal_noise_ratio':signal_noise_ratio, "defect": target}]
    if st.button('Append data', key='append'):
        URL = 'http://defect_classify:5001/append_data'
        DATA = {'data_path':data_path, 'data':sample_append}
        DATAAPPEND_RESPONSE = requests.post(url = URL, json = DATA)
        st.info("Appending new data point")
        st.info(DATAAPPEND_RESPONSE.text)
        if len(DATAAPPEND_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
            st.info(DATAAPPEND_RESPONSE.text)
        else:
            st.success(str(DATAAPPEND_RESPONSE.json()))#.get('Maintenance Recommendation')))
# Help tab frontend below
    
with help_tab:
    st.markdown("#### Input Descriptions for Ultrasonic NDT Testing:")
    st.markdown("- Backwall: Binary feature indicating the presence (1) or absence (0) of a back wall signal.")
    st.markdown("- Frontwall: Binary feature indicating the presence (1) or absence (0) of a front wall signal.")
    st.markdown("- Ramp: Binary feature indicating the presence (1) or absence (0) of a ramp signal.")
    st.markdown("- Geometry: Binary feature indicating the presence (1) or absence (0) of geometry-related signal.")
    st.markdown("- Number of Peaks: Integer feature representing the number of peaks in the ultrasonic signal.")
    st.markdown("- Noise: Continuous feature representing the noise level in the ultrasonic signal.")
    st.markdown("- Maximum Value: Continuous feature representing the maximum value in the ultrasonic signal.")
    st.markdown("- Minimum Value: Continuous feature representing the minimum value in the ultrasonic signal.")
    st.markdown("- Signal-to-Noise Ratio: Continuous feature representing the signal-to-noise ratio in the ultrasonic signal.")
    st.markdown("- Defect: Target label indicating the defect class (0, 1, or 2) detected during ultrasonic NDT testing.")
    st.markdown("#### Code Samples:")
    
    st.markdown("##### Conversion of XGBoost to Daal4py Model")
    daalxgboost_code = '''xgb_model = xgb.train(self.parameters, xgb_train, num_boost_round=100)
        self.d4p_model = d4p.get_gbt_model_from_xgboost(xgb_model)'''
    st.code(daalxgboost_code, language='python')
    
    st.markdown("##### Inference with Daal4py Model")
    daalxgboost_code = '''
    daal_predict_algo = d4p.gbt_classification_prediction(
            nClasses=num_class,
            resultsToEvaluate="computeClassLabels",
            fptype='float')
            
    daal_prediction = daal_predict_algo.compute(data, daal_model)
    '''
    st.code(daalxgboost_code, language='python')
    st.markdown(
        """
        #### Help

        This app is still under development. Please contact the github repo owner for more details.
        """
    )
