import joblib
import os
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from data_utils import synthetic_defects, fetch_coordinates_data

# Depreciated - this is a test function now
def get_data(x_coord, y_coord):
    # set random seed
    np.random.seed(x_coord + y_coord*10000)
    arr = np.random.randint(low=0, high=20, size=(1, 10))
    return arr
class WaveInference():
    def __init__(self,model_name: str,  model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        #self.data_path = data_path
        self.data = ''
        self.dataset, self.labels = '', ''
        self.prediction = ''
        self.status = ''
        
    def load_model_and_data(self, n_channels, img_dim, test_scan):
        self.img_dim = img_dim
        self.n_channels = n_channels
        self.test_scan = test_scan
 # load daal model
        self.data, self.labels, self.pixelmap, self.columns = synthetic_defects(self.img_dim, self.n_channels, self.test_scan, stat_features=False)
        #print("dataloaded")
        #print(self.data)
        model_file_path = os.path.join(self.model_path,self.model_name+".joblib")
        with open(model_file_path, "rb") as fh:
            self.model = joblib.load(fh.name)
            print("model  and data loaded")
        # if self.data_path != '':    
        #     with open(self.data_path, "rb") as fh:
        #         self.dataset = joblib.load(fh.name)
        # else:
        #     self.dataset = np.random.randint(low=0, high=20, size=(1000, 10))
        #     self.labels = np.random.randint(low=0, high=2, size=(1000, 1))    
            
    def inference(self, x_coord, y_coord):
        self.x_coord = x_coord
        self.y_coord = y_coord
        #self.data = get_data(x_coord, y_coord)
        self.datapoint = fetch_coordinates_data(data = self.data, pixelmap= self.pixelmap, x_coord=self.x_coord, y_coord=self.y_coord, stat_features=False)        
        self.prediction=3
        #self.prediction = self.model.predict(self.datapoint)
        self.status = 'unknown defect'
        print("In inference function")
        if self.prediction == 0:
            self.status = 'No defect detected'
        elif self.prediction == 1:
            self.status = 'Defect detected'
        elif self.prediction == 2:
            self.status = ''
        # results, and wavetoplot
        # next step is to plot the waveform    
        return self.status, self.datapoint.values

    def lime_explanation(self):
        print("Entering lime explanation")
        # shape should be (n_features,)
        print(self.data.shape)
        print(self.labels)
        print(self.data)
        print("shape of data : ", self.data.shape)
        #data = get_data(x_coord, y_coord)[0]
        datapoint = fetch_coordinates_data(data= self.data, 
                                           pixelmap= self.pixelmap, 
                                           x_coord=self.x_coord, 
                                           y_coord=self.y_coord, 
                                           stat_features=False)
        print(datapoint)
        print("generating explainer")
        print(self.model.predict_proba)
        print("labels shape is  : ", self.labels.shape)
        self.labels = self.labels.values.reshape(-1)
        explainer = LimeTabularExplainer(self.data.values,  
                                         training_labels = self.labels, 
                                         mode="classification")
        print("explainer generated")
        print("generating explanation")
        print("prediction")
        print(self.model.predict_proba(datapoint.values))
        print(datapoint.values.shape)
        print(datapoint.values)
        explanation = explainer.explain_instance(datapoint.values.reshape(-1,), 
                                                 self.model.predict_proba, 
                                                 num_features=len(self.columns))
        print("Explanation generated")
        print(explanation)
        explanation = explanation.as_list()
        print("explanation generated")
        feature_importance = [b for a, b in explanation]
        print(f"feature importance : {feature_importance}")
        return feature_importance