import os
import json
import pandas as pd
import streamlit as st
from PIL import Image
import requests

# Set the title of the app
st.title('Defect class identification and categorization')

# Create two tabs: "Application" and "Help"
app_tab, help_tab = st.tabs(["Application", "Help"])

with app_tab:
    ####################PART-1 MODEL TRAINING #################################
    # "st.session_state object:" , st.session_state
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
    st.markdown(
        """
        #### Model training for defect classification.
        """
    )
    # Input data
    #with st.expander("Input Data"): 
    cola1, cola2 = st.columns(2)
    with cola1:
        test_scan = st.selectbox('Test Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan", "random"], placeholder="low_defect_scan")
        train_scan = st.selectbox('Train Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan", "random"], placeholder="high_defect_scan")
        append_path = st.text_input('Training Append Data File Path (If you appended new data below)', key='data', value='./box/datasets/defect_classify/data_train.csv') 
        model_path = st.text_input('Model Save Path', key='model path', value='./box/models/defect_classify/')
        
    with cola2:    
        n_channels = st.slider('Number of channels in waveform',min_value=1, max_value=128, value=10, step=1)
        img_dim = st.selectbox('Image Dimension', [16,32,64,128], placeholder="64")
        model_name = st.text_input('Model Name',key='model name', help='The name of the model (change when re-training)', value='model')
        with st.expander("More info on data"):
            st.info("""
                   ###### Recomended : Choose test_scan and train_scan with low, medium and high defect scans.
                   ######              These scans are persisted through the three phases, so you can choose the same scans for all phases.
                   ######             (It is advised to choose high_defect_scan for training and others for testing (due to data imbalance in low/medium defect scans)")
                   ###### Choose the following data file if you loaded csv prior to running docker ./box/datasets/defect_classify/train.csv
                    """)
        
    # Button to train the model
    if st.button('Train Model', key='training'):
        # Build the request
        URL = 'http://defect_classify:5001/train'
    
        DATA = {
                'img_dim':img_dim, 
                'n_channels':n_channels,
                'test_scan':test_scan,
                'train_scan':train_scan,
                'model_name':model_name, # 'model
                'model_path':model_path, 
                'append_path' : append_path,
                'ncpu': 1} # 
        
        print(DATA)
        TRAINING_RESPONSE = requests.post(url=URL, json=DATA)
        #print(TRAINING_RESPONSE)
        #st.info(TRAINING_RESPONSE)
        # Check the response status code
        if len(TRAINING_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
            st.info(TRAINING_RESPONSE.text)
        else:
            st.success('Training was Succesful')
         #   st.info(TRAINING_RESPONSE.text)
            st.info("Model trained on {} datapoints".format(str(TRAINING_RESPONSE.json().get('xtrain_len'))) )
            st.info('Model Validation Accuracy Score: ' + str(TRAINING_RESPONSE.json().get('validation scores')))

    ####################PART-2 : Partial dependence visualization##############################
    
    st.divider()
    st.markdown(
        """
        #### Partial dependence analysis of a feature on outcome.
        """
    )
    selected_model_path = st.text_input('Selected Model Path',key='model path selection', value='./box/models/defect_classify/model.joblib')
    col21, col22, col23 = st.columns(3)
    fetched_data = None 
    
    def update_data(data):
        dfcols = {"qc1": "Quality Check-1", "qc2": "Quality Check-2", "qc3": "Quality Check-3", "qc4": "Quality Check-4", "min": "Minimum Value", "max": "Maximum Value", "mean": "Mean", "std": "Standard Deviation", "snr": "Signal-to-Noise Ratio", "num_peaks": "Number of Peaks"}
        df = pd.DataFrame(data)
        df.rename(columns=dfcols, inplace=True) 
        df = df.T
        df.reset_index(inplace=True)
        df.columns = ["Feature", "Value"]
       # df.columns = ["Feature", "Value"]
       # st.table(df)
        df = df.round(3).astype(object)
        df.style.background_gradient(axis=None, vmin=1, vmax=5, cmap="YlGnBu")
        cell_hover = {  # for row hover use <tr> instead of <td>
        'selector': 'td:hover',
        'props': [('background-color', '#ffffb3')]
        }
        index_names = {
        'selector': '.index_name',
        'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
        }
        headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #FFFDF4BF; color: #FF3399FF;'
        }
        df = df.style.set_table_styles([cell_hover, index_names, headers])
        df = df.set_table_styles([
            {'selector': 'th.col_heading', 'props': 'text-align: center;'},
            {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5em;'},
            {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'},
        ], overwrite=False)
        st.table(df)
        # st.session_state.min = data['min']
        # st.session_state.max = data['max']
        # st.session_state.mean = data['mean']
        # st.session_state.snr = data['snr']
        # st.session_state.std = data['std']
        # st.session_state.num_peaks = data['num_peaks']
        # st.session_state.qc1 = data['qc1']
        # st.session_state.qc2 = data['qc2']
        # st.session_state.qc3 = data['qc3']
        # st.session_state.qc4 = data['qc4']
                
    with col21:
        st.info("Select from scan?")
        #st.info("X and Y Coordinates from the original scan")
        x_coord = st.slider('X Coordinate', min_value=0, max_value=512, step=1, value=5)
        y_coord = st.slider('Y Coordinate', min_value=0, max_value=512, step=1, value=5)
        
        if st.button('Fetch data', key='fetch_data'):
            URL = 'http://defect_classify:5001/fetch_coordinates_data'
            DATA = {'img_dim': img_dim, 'n_channels': n_channels, 'test_scan': test_scan, 'x_coord':x_coord, 'y_coord':y_coord}
            DATA_RESPONSE = requests.post(url = URL, json = DATA)
           
            #st.info(DATA_RESPONSE.text)
            if DATA_RESPONSE.text != None:
                data = json.loads(DATA_RESPONSE.text)["fetched_data"]#[0]
                update_data(data = data)
            else:
                st.info("Coordinates out of range")
                
                update_data(DATA_RESPONSE.text)
            # Now call the update function from here to update sample data    
            #st.info()
        with st.expander("More info on quality checks"):
            st.info("""
                   ###### Quality checks are binary features indicating no issue or indcating presence of an issue.
                   ###### Please refer to Help tab for a case-based description of a case-based example for quality checks. 
                    """)    
    with col22:
        st.info("Inspector's Input")
        qc1 = st.selectbox('Quality Check-1', [0,1], key='qc1')
        qc2 = st.selectbox('Quality Check-2', [0,1], key='qc2')
        qc3 = st.selectbox('Quality Check-3', [0,1], key='qc3')
        qc4 = st.selectbox('Quality Check-4', [0,1], key='qc4')

    with col23:
        st.info("waveform characteristics")
        min = st.slider('Minimum Value', min_value=0.00, max_value=10.00, step=.05, value=5.00, key='min')
        max = st.slider('Maximum Value', min_value=0.00, max_value=10.00, step=.05, value=5.00  , key='max')
        mean = st.slider('Mean', min_value=0.00, max_value=10.00, step=.05, value=5.00, key='mean')
        snr = st.slider('Signal-to-Noise Ratio', min_value=0.00, max_value=10.00, step=.05, value=5.00, key='snr')
        std = st.slider('Standard Deviation', min_value=0.00, max_value=10.00, step=.05, value=5.00, key='std')
        num_peaks = st.slider('Number of Peaks', min_value=0, max_value=100, step=1, value=5, key='num_peaks')
    
    sample = [{'qc1':qc1, 'qc2':qc2, 'qc3':qc3, 'qc4':qc4, 'min':min, 'max':max, 'mean':mean, 'std':snr, 'snr': std,'num_peaks':num_peaks}]

    if st.button('Defect Categorization', key='analysis'):
        URL = 'http://defect_classify:5001/predict'
        DATA = {'data':sample, 'model_path':model_path, 'model_name':model_name, 'num_class':3, 'scaler': True}
        INFERENCE_RESPONSE = requests.post(url = URL, json = DATA)
        if len(INFERENCE_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
            st.info(INFERENCE_RESPONSE.text)
        else:
            st.success(str(INFERENCE_RESPONSE.json().get('Defect Result')))

    data_path ="./box/datasets/defect_classify/train.csv"
    #For appending data    
    st.markdown("""
               #### Generated a new relavant data point? and want to append it to the training data?
                """)
    DEFECT_TYPES = {0:'No Defect', 1:'Type-1 Defect', 2:'Type-2 Defect'}
    target = st.selectbox('Select defect category before append', [0,1,2], format_func=lambda x: DEFECT_TYPES[x])
    #sample_append = [{'qc1':backwall, 'frontwall':frontwall, 'ramp':ramp, 'geometry':geometry, 'no_peaks':no_peaks, 'noise':noise, 'max':max, 'min':min, 'signal_noise_ratio':signal_noise_ratio, "defect": target}]
    sample_append = [{'qc1':qc1, 'qc2':qc2, 'qc3':qc3, 'qc4':qc4, 'min':min, 'max':max, 'mean':mean, 'std':std,'snr':snr, 'num_peaks':num_peaks, 'defect':target}]
    #sample_append = [qc1, qc2, qc3, qc4, min, max, mean, snr, num_peaks, target]
    
    if st.button('Append data', key='appending'):
        URL = 'http://defect_classify:5001/append_data'
        DATA = {'data_path':append_path, 'data': sample_append}
        
        DATAAPPEND_RESPONSE = requests.post(url = URL, json = DATA)
        st.info("Appending new data point")
        
        if len(DATAAPPEND_RESPONSE.text) < 40:       
            st.error("Data appending failed")
            st.info(DATAAPPEND_RESPONSE.text)
        else:
            st.success("Data appended successfully")
            st.info(DATAAPPEND_RESPONSE.json().get('msg'))
# Help tab frontend below
############################### HELP TAB#################################
with help_tab:
    st.markdown("#### Input Descriptions for Ultrasonic NDT Testing:")
    st.markdown("### Quality Checks: 1 and 2")
    st.markdown("""A waveform signal as seen in Phase-2 comprises of many peaks. 
                The presence or absense of the peaks tells us if the ultransonic signal is able to detect 
                the front and the back surface of a given material. 
                Below we characterize these surfaces as frontwall and backwall which correspond to the front and back surfaces, respectively.""")
    st.markdown("- Quality Check-1 (Backwall): Binary feature indicating the presence (1) or absence (0) of a back wall signal.")
    st.markdown("- Frontwall: Binary feature indicating the presence (1) or absence (0) of a front wall signal.")
    st.markdown("#### Quality Checks: 3 and 4")
    st.markdown("""
                Often times a given surfaces has some geometry to it. For example, assume we are scanning the door of a car.
                The door has a handle and a lock. The handle and the lock are protruding features on the door. 
                Therby they produce an artifact in the waveform which can be treaded as a geometry. Second, think of the thickness changess in the metal, the presence of these shifts in thicknesses also causes certain artifacts in the ultrasonic waveform which are referred to as "ramp".  
                """)
    st.markdown("- Ramp: Binary feature indicating the presence (1) or absence (0) of a ramp signal.")
    st.markdown("- Geometry: Binary feature indicating the presence (1) or absence (0) of geometry-related signal.")
    st.markdown("####Features generated through feature engineering on raw waveform:")
    st.markdown("- Number of Peaks: Integer feature representing the number of peaks in the ultrasonic signal.")
    st.markdown("- Noise: Continuous feature representing the noise level in the ultrasonic signal.")
    st.markdown("- Maximum Value: Continuous feature representing the maximum value in the ultrasonic signal.")
    st.markdown("- Minimum Value: Continuous feature representing the minimum value in the ultrasonic signal.")
    st.markdown("- Signal-to-Noise Ratio: Continuous feature representing the signal-to-noise ratio in the ultrasonic signal.")
    st.markdown("- Defect: Target label indicating the defect class (0, 1, or 2) detected during ultrasonic NDT testing.")
    st.markdown("#### Code Samples:")
    
    st.markdown(
        """
        #### Help

        This app is still under development. Please contact the github repo owner for more details.
        """
    )