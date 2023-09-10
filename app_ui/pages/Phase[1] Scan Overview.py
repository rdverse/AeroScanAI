import streamlit as st
from PIL import Image
import requests
import ast
import pandas as pd
import numpy as np
# Set the title of the app
st.title('Scan Anomaly Detection')

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
                    ###### 3D-UNet model with adaptive input training
                    """)
    with col22:
        st.markdown("""
            ##### Phase 1 : Train a model using 3D scans with defects and without defects.
            This gives an overview of the data and the areas that might potentially have a defect.   
            """)
        
    cola1, cola2 = st.columns(2)
    
    with cola1:
        st.info("Model Parameters")
        
        model_name = st.text_input('Model Name',key='model name', help='The name of the model (change when re-training)', value='model')
        model_path = st.text_input('Model Save Path', key='model path', value='./box/models/scan_anomaly/')
        #test_scan = st.selectbox('Test Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan", "random"], placeholder="low_defect_scan")
        st.info("Training Parameters")
        n_cpus = st.slider('Number of CPUs',min_value=1, max_value=128, value=1, step=1)
        n_epochs = st.slider('Number of Epochs',min_value=1, max_value=100, value=3, step=1)
        batch_size = st.slider('Batch Size',min_value=1, max_value=128, value=32, step=1)

    with cola2:    
        st.info("Data Parameters")
        n_channels = st.slider('Number of channels in waveform',min_value=1, max_value=512, value=10, step=1)
        n_classes = st.selectbox('Number of classes', [1], placeholder="1")
        n_samples = st.slider('Number of samples',min_value=100, max_value=10000, value=100, step=100)
        img_dim = st.selectbox('Image Dimension', [16,32,64,128,256,512], placeholder="64")
        percent_test = st.slider('Percentage of data saved for Testing',min_value=0.1, max_value=0.5, value=0.3, step=0.1)
        #with st.expander("More info on data"):
            
    # Button to train the model
    if st.button('Train Model', key='training'):
        # Build the request
        URL = 'http://scan_anomaly:5003/train'
    
        DATA = { 'img_dim':img_dim, 
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
        
    test_scan = st.selectbox('Test Scan', ["low_defect_scan", "medium_defect_scan","high_defect_scan", "random"], placeholder="low_defect_scan")
    if st.button('Visualize', key='predict_visualize'):
        URL = 'http://scan_anomaly:5003/predict'
        DATA = {'model_name': model_name, 
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
            preds = np.array(ast.literal_eval(INFERENCE_RESPONSE.json().get('preds')))
            labels = np.array(ast.literal_eval(INFERENCE_RESPONSE.json().get('labels')))
            inputs = np.array(ast.literal_eval(INFERENCE_RESPONSE.json().get('inputs')))
            st.image(preds)
            #st.image(labels)
            #st.image(inputs)
            #st.info('Model Validation Accuracy Score: ' + str(TRAINING_RESPONSE.json().get('validation scores')))
            
#     # Separator
#     st.divider()
#     st.markdown(
#         """
#         #### Waveform defect prediction and interpretation using explainable-AI.
#         """
#     )

#     model_name = st.text_input('Selected Model Name',key='model name selection', value='model')
#     model_path = st.text_input('Selected Model Path',key='model path selection', value='./box/models/scan_anomaly/')
#     col21, col22, col23 = st.columns(3)

#     with col21:
#         st.info("Inspector's Input")
#         x_coord = st.number_input("x coordinate", min_value=0, max_value=128, step=1, value=10)
#     with col22:
#         st.info("Enter Y coordinate")
#         y_coord = st.number_input('y coordinate', min_value=0, max_value=128, step=1, value=10)
#     #sample = [{'x_coord':10, 'y_coord':10}]
#     sample = [{'x_coord':x_coord, 'y_coord':y_coord}]
    
#     if st.button('Analyze waveform', key='analysis'):
#         URL = 'http://scan_anomaly:5003/predict'
#         DATA = {'model_name': model_name, 
#                 'model_path':model_path,
#                 'n_channels': n_channels, 
#                 'img_dim': img_dim, 
#                 'test_scan' : test_scan, 
#                 'x_coord':x_coord , 
#                 'y_coord' : y_coord ,
#                 'num_class':2}
#         INFERENCE_RESPONSE = requests.post(url = URL, json = DATA)
#         #print(INFERENCE_RESPONSE.json())
#         st.info(INFERENCE_RESPONSE.text)
#         ####################uncomment
#         import ast
#         st.info(type(INFERENCE_RESPONSE.json().get('wavetoplot')))
#         data = ast.literal_eval(INFERENCE_RESPONSE.json().get('wavetoplot'))
#         feature_importance = ast.literal_eval(INFERENCE_RESPONSE.json().get('feature_importance'))
#         data, feature_importance, time = np.array(data).reshape(-1,1), np.array(feature_importance).reshape(-1,1), np.arange(len(data)).reshape(-1,1)
#         dataImpDF = pd.DataFrame(np.hstack((data, feature_importance, time)), columns = ['Amplitude', 'Feature Importance', 'Time'])
        
#         st.line_chart(dataImpDF, 
#                       y = 'Amplitude',
#                       x = 'Time')
#         st.bar_chart(dataImpDF, 
#                      y = 'Feature Importance',
#                      x = 'Time')
#         with st.expander('More info on feature importance'):
#             st.info("""
#                     ###### Feature Importance
#                     Here we use LIME (local agnostic model explanation) to explain the model's prediction for a given input (i.e., the raw waveform).
#                     """)
#        ################## 
#         # print(INFERENCE_RESPONSE)
#         # st.info(INFERENCE_RESPONSE)
#         # st.info(INFERENCE_RESPONSE.text)
#         # if len(INFERENCE_RESPONSE.text) < 40:       
#         #     st.error("Inference Failed")
#         #     st.info(INFERENCE_RESPONSE.text)
#         # else:
#         #     print(INFERENCE_RESPONSE)#.json().get('wavetoplot')
#         #     #st.success(str(INFERENCE_RESPONSE.json().get('results')))
#         #     #st.line_chart()
    
# with help_tab:
#     st.markdown("#### Importance of Waveform Probing")
#     st.markdown(        """
#         #### Why is it important to analyze the waveform of each pixel in a 3D scan?
#         #####  The waveform of each pixel in a 3D scan has immense information about the defect or a normal signal. 
#         ##### Therefore, we first train an random forest model and pick the optimal configuration using grid search.
#         ##### Next, we use explainable-AI such as LIME to interpret the model's prediction for a given input (i.e., the raw waveform).
#         ##### This helps us understand what parts of the waveform were given most importance for the prediction.
#         ##### To interpret this plot, .
#     """)