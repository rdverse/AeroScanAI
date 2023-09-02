import streamlit as st
from PIL import Image
import requests

# Set the title of the app
st.title('Patient Readmission Prediction')

# Create two tabs: "Application" and "Help"
app_tab, help_tab = st.tabs(["Application", "Help"])

# "Application" tab

with app_tab:

    # Header image
    col11, col22 = st.columns(2)

    with col11:
        image = Image.open('./assets/qaqc.png')
        st.image(image)
        
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
        #### Patient Readmission Prediction Model Training
        """
    )

    # Input data
    data_file = st.text_input('Training Data File Path', key='data', value='./store/datasets/patient_readmission/train.csv')
    window_size = st.number_input('Window', min_value=50, max_value=200, value=125, step=5)
    lag_size = st.slider('Lagging Window', min_value=5, max_value=50, value=25, step=5)
    epochs = st.number_input('Epochs', min_value=1, max_value=100, step=1, value=5)
    batch_size = st.number_input('Batch Size', min_value=100, max_value=1000, step=1, value=512)
    model_path = st.text_input('Model Save Path', key='model path', value='./store/models/patient_readmission/model.pb')
    test_size = st.slider('Percentage of data saved for Testing', min_value=0.10, max_value=0.90, value=0.30, step=0.05)

    # Button to train the model
    if st.button('Train Model', key='training'):
        # Build the request
        URL = 'http://patient_readmission:5001/train'
        DATA = {
            'filepath': data_file,
            'window': window_size,
            'lag_size': lag_size,
            'test_size': test_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'save_model_dir': model_path
        }
        TRAINING_RESPONSE = requests.post(url=URL, json=DATA)

        # Check the response status code
        if TRAINING_RESPONSE.status_code == 200:
            st.success('Training was successful!')
        else:
            st.error('Training failed!')
            st.info(TRAINING_RESPONSE.text)

    # Separator
    st.divider()

    # Patient readmission prediction analysis

    st.markdown(
        """
        #### Patient Readmission Prediction Analysis
        """
    )

    # Inference inputs
    selected_model_path = st.text_input('Selected Model Path', key='demand forecaster model', value='./store/models/patient_readmission/model.pb')
    analysis_save_path = st.text_input('Forecast Demand Analysis Save Path', value='./store/outputs/patient_readmission/')
    input_data = st.text_input('Input Data Path', value='./store/datasets/patient_readmission/test.csv')
    inf_window_size = st.number_input('Window', min_value=50, max_value=200, value=125, step=5, key='demand forecaster window')
    inf_lag_size = st.slider('Lagging Window', min_value=5, max_value=50, value=25, step=5, key='demand forecaster lag window')
    inf_batch_size = st.number_input('Batch Size', min_value=100, max_value=1000, step=1, value=512, key='demand forecaster batch')
    interations = st.number_input('Interations', min_value=50, max_value=200, value=100, step=5, key='demand forecaster iterations')

    # Button to run the inference
    if st.button('Run Patient Readmission Prediction Analysis', key='analysis'):
        # Build the request
        URL = 'http://patient_readmission:5001/predict'
        DATA = {
            'keras_saved_model_dir': selected_model_path,
            'output_saved_dir': selected_model_path,
            'input_file': input_data,
            'results_save_dir': analysis_save_path,
            'window': inf_window_size,
            'lag_size': inf_lag_size,
            'batch_size': inf_batch_size,
            'num_iters': interations
        }
        INFERENCE_RESPONSE = requests.post(url=URL, json=DATA)

        # Check the response status code
        if INFERENCE_RESPONSE.status_code == 200:
            st.success('Analysis was successful!')
            st.info(INFERENCE_RESPONSE.text)
        else:
            st.error('Analysis failed!')
            st.info(INFERENCE_RESPONSE.text)

# "Help" tab

with help_tab:
    st.markdown(
        """
        #### Help

        This app is still under development. Please contact the developers for more information.
        """
    )
