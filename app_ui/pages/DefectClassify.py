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
            ##### This app predicts the probability of a patient being readmitted to the hospital within a certain period of time after discharge.
            It uses a machine learning model that is trained on historical medical data.
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
                  'test_size': test_size, 'ncpu': 4}
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

    # default
    manufacturer_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    model_list = ['Gen1', 'Gen2', 'Gen3', 'Gen4']
    lubrication_type_list = ['LTA', 'LTB', 'LTC']
    product_assignment_list = ['PillA', 'PillB', 'PillC']

    with col21:
        manufacturer = st.selectbox('Manufacturer', manufacturer_list)
        generation = st.selectbox('Generation', model_list)
        age = st.number_input('Robot Age', min_value=0, max_value=25, step=1, value=0)

    with col22:
        temperature = st.number_input('Temperature', min_value=50, max_value=300, step=1)
        motor_current = st.number_input('Motor Current', min_value=0.00, max_value=10.00, step=.05, value=5.00)
        lubrication_type = st.selectbox('Lubrication Type', lubrication_type_list)
    with col23:
        last_maintenance = st.number_input('Last Maintenance', min_value=0, max_value=60, step=1)
        num_repairs = st.number_input('Repair Counts', min_value=0, max_value=50, step=1)
        product_assignment = st.selectbox('Pill Product Assignment', product_assignment_list)
        
        
    sample = [{'Age':age, 'Temperature':temperature, 'Last_Maintenance':last_maintenance, 'Motor_Current':motor_current,
       'Number_Repairs':num_repairs, 'Manufacturer':manufacturer, 
       'Generation':generation,'Lubrication':lubrication_type, 'Product_Assignment':product_assignment}]

    if st.button('Run Maintenance Analysis', key='analysis'):
        URL = 'http://defect_classify:5001/predict'
        DATA = {'sample':sample, 'model':model_path, 'num_class':3}
        INFERENCE_RESPONSE = requests.post(url = URL, json = DATA)
        st.info("another trail")
        st.info(INFERENCE_RESPONSE)
        st.info(INFERENCE_RESPONSE.text)
        if len(INFERENCE_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
            st.info(INFERENCE_RESPONSE.text)
        else:
            st.success(str(INFERENCE_RESPONSE.json().get('Maintenance Recommendation')))
            
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
