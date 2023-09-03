import streamlit as st
from PIL import Image
import requests

# Set the title of the app
st.title('AIrcrafy defect classification')

# Create two tabs: "Application" and "Help"
app_tab, help_tab = st.tabs(["Application", "Help"])

# "Application" tab

with app_tab:
    # Header image
    col11, col22 = st.columns(2)
    with col11:
        #image = Image.open('./assets/qaqc.png')
        #st.image(image)
        st.markdown("text")
    with col22:
        st.markdown(
            """
            ##### Detect defect in fuselages
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
    data_file = st.text_input('Training Data File Path', key='data', value='./box/datasets/defect_classify/train.csv')
    window_size = st.number_input('Window', min_value=50, max_value=200, value=125, step=5)
    lag_size = st.slider('Lagging Window', min_value=5, max_value=50, value=25, step=5)
    epochs = st.number_input('Epochs', min_value=1, max_value=100, step=1, value=5)
    batch_size = st.number_input('Batch Size', min_value=100, max_value=1000, step=1, value=512)
    model_path = st.text_input('Model Save Path', key='model path', value='./box/models/defect_classify/')
    model_name = st.text_input('Model Name',key='model name', help='The name of the model without extensions', value='model')
    test_size = st.slider('Percentage of data saved for Testing',min_value=5, max_value=50, value=25, step=5)

    # Button to train the model
    if st.button('Train Model', key='training'):
        # Build the request
        URL = 'http://defect_classify:5001/train'
    
        DATA = {'file':data_file, 'model_name':model_name, 'model_path':model_path, 
                  'test_size': test_size, 'ncpu': 1}
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
