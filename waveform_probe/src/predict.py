import joblib
import os
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


def get_data(x_coord, y_coord):
    # set random seed
    np.random.seed(x_coord + y_coord*10000)
    arr = np.random.randint(low=0, high=20, size=(1, 10))
    return arr

class WaveInference():
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.data = ''
        self.dataset, self.labels = '', ''
        self.prediction = ''
        self.status = ''
        
    def load_model_and_data(self ):
        # load daal model
        with open(self.model_path, "rb") as fh:
            self.model = joblib.load(fh.name)
        if self.data_path != '':    
            with open(self.data_path, "rb") as fh:
                self.dataset = joblib.load(fh.name)
        else:
            self.dataset = np.random.randint(low=0, high=20, size=(1000, 10))
            self.labels = np.random.randint(low=0, high=2, size=(1000, 1))    
            
    def inference(self, x_coord, y_coord):
        self.data = get_data(x_coord, y_coord)
        self.prediction = self.model.predict(self.data)
        self.status = 'unknown defect'
        
        if self.prediction == 0:
            self.status = 'No defect detected'
        elif self.prediction == 1:
            self.status = 'Potentially a defect'
        elif self.prediction == 2:
            self.status = ''
        # results, and wavetoplot
        # next step is to plot the waveform    
        return self.status, self.data

    def lime_explanation(self, x_coord, y_coord):
        # shape should be (n_features,)
        print(self.dataset.shape)
        print(self.labels)
        data = get_data(x_coord, y_coord)[0]
        print(data)
        print("generating explainer")
        explainer = LimeTabularExplainer(self.dataset,  training_labels = self.labels, mode="classification")
        print("explainer generated")
        print("generating explanation")
        print(self.model.predict_proba)
        print("prediction")
        print(self.model.predict_proba(data.reshape(1, -1)))
        explanation = explainer.explain_instance(data, self.model.predict_proba).as_list()
        print("explanation generated")
        feature_importance = [b for a, b in explanation]
        print(f"feature importance : {feature_importance}")
        return feature_importance
        
# def inference(model_path, x_coord, y_coord, num_class: int = 2):
#     # Use .joblib extension to save file
#     # RF is robust with unscaled features    
#     print("entered inference")
#     data = get_data(x_coord, y_coord)
    
#     if "model.joblib" not in model_path:
#         model_path = os.path.join(model_path,"model.joblib")
         
#     # load daal model
#     with open(model_path, "rb") as fh:
#         model = joblib.load(fh.name)
    
#     prediction = model.predict(data)
#     print(prediction)
#     print(data)
#     status = 'unknown defect'
    
#     if prediction == 0:
#         status = 'No defect detected'
#     elif prediction == 1:
#         status = 'Potentially a defect'
#     elif prediction == 2:
#         status = ''
#     # results, and wavetoplot
#     # next step is to plot the waveform    
#     return status, data
    


# def append_data(data_path, data):
#     """
#     Append new data to an existing or new pickle file.

#     Parameters
#     ----------
#     data_path : str
#         The file path to the pickle file where data will be stored or appended.
#     data : pd.DataFrame
#         The DataFrame containing the new data to be appended.

#     Returns
#     -------
#     int
#         The total number of records in the combined DataFrame after appending.

#     Notes
#     -----
#     If the specified `data_path` does not exist, a new pickle file will be created.
#     If the file already exists, the new data will be appended to the existing data.
#     """
#     combined_data = pd.DataFrame()
#     previousdatalen = 0
#     # Check if the data file already exists
#     if os.path.exists(data_path):
#         # Read the existing pickle file into a DataFrame
#         existing_data = pd.read_pickle(data_path)
#         previousdatalen = len(existing_data)
#     else:
#         existing_data = None

#     if existing_data is not None:
#         # Append the new data to the existing data
#         combined_data = pd.concat([existing_data, data], ignore_index=True)
#         # Save the combined data back to the pickle file
#         combined_data.to_pickle(data_path)
#         print("Data appended to existing pickle file.")
#     else:
#         # If no existing data found, create a new pickle file
#         data.to_pickle(data_path)
#         print("New data file created.")

#     print("Data saved to pickle file.")
#     return {"msg": f"Data points before appending : {previousdatalen}, and data points after appending: {len(combined_data)}"}