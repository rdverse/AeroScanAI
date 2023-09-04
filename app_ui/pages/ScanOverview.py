import streamlit as st
from PIL import Image
import requests

# Set the title of the app
st.title('Waveform Probing')

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
    st.markdown(
        """
        #### Aero defect classification model training
        """
    )
    # Input data
    data_file = st.text_input('Training Data File Path', key='data', value='./box/datasets/scan_anomaly/train.csv')
    model_path = st.text_input('Model Save Path', key='model path', value='./box/models/scan_anomaly/')
    model_name = st.text_input('Model Name',key='model name', help='The name of the model without extensions', value='model')
    test_size = st.slider('Percentage of data saved for Testing',min_value=5, max_value=50, value=25, step=5)

    # Button to train the model
    if st.button('Train Model', key='training'):
        # Build the request
        URL = 'http://scan_anomaly:5002/train'
    
        DATA = {'file':data_file, 
                'model_name':model_name, 
                'model_path':model_path, 
                'test_size': test_size, 
                'ncpu': 1}
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
    st.markdown(
        """
        #### Aero defect classification analysis
        """
    )

    st.markdown('#### Predictive Maintenance Analysis')
    
    selected_model_path = st.text_input('Selected Model Path',key='model path selection', value='./box/models/scan_anomaly/model.joblib')
    selected_data_path = st.text_input('Selected Data Path',key='data path selection', value='')
    col21, col22, col23 = st.columns(3)

    with col21:
        st.info("Inspector's Input")
        x_coord = st.number_input("x coordinate", min_value=0, max_value=128, step=1, value=10)
    with col22:
        st.info("Enter Y coordinate")
        y_coord = st.number_input('y coordinate', min_value=0, max_value=128, step=1, value=10)
    #sample = [{'x_coord':10, 'y_coord':10}]
    sample = [{'x_coord':x_coord, 'y_coord':y_coord}]
    
    if st.button('Run Maintenance Analysis', key='analysis'):
        URL = 'http://scan_anomaly:5002/predict'
        DATA = {'model_path':selected_model_path, 'data_path' : selected_data_path, 'x_coord':x_coord , 'y_coord' : y_coord ,'num_class':3}
        INFERENCE_RESPONSE = requests.post(url = URL, json = DATA)
        #print(INFERENCE_RESPONSE.json())
        st.info(INFERENCE_RESPONSE.text)
        import ast
        st.info(type(INFERENCE_RESPONSE.json().get('wavetoplot')))
        data = ast.literal_eval(INFERENCE_RESPONSE.json().get('wavetoplot'))
        feature_importance = ast.literal_eval(INFERENCE_RESPONSE.json().get('feature_importance'))
        st.info(data)
        st.line_chart(data)
        st.bar_chart(feature_importance)
        # print(INFERENCE_RESPONSE)
        # st.info(INFERENCE_RESPONSE)
        # st.info(INFERENCE_RESPONSE.text)
        # if len(INFERENCE_RESPONSE.text) < 40:       
        #     st.error("Inference Failed")
        #     st.info(INFERENCE_RESPONSE.text)
        # else:
        #     print(INFERENCE_RESPONSE)#.json().get('wavetoplot')
        #     #st.success(str(INFERENCE_RESPONSE.json().get('results')))
        #     #st.line_chart()
    
with help_tab:
    st.markdown("#### Input Descriptions for Ultrasonic NDT Testing:")
    st.markdown(        """
        #### Help
        This app is still under development. Please contact the github repo owner for more details."""
    )