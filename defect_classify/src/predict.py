import daal4py as d4p
import joblib
import os
# import pdb
import numpy as np
import pandas as pd

def inference(model_name, model_path, data, num_class: int = 3, scaler: bool = True):
    # Use .joblib extension to save file
    scaler_path = ""
    if "model" in model_path:
        scaler_path = scaler_path.replace("model", "")
    if ".pb" in model_path:
        scaler_path = "".join(model_path.split(".pb")) + model_name + "_scaler.joblib"
    elif ".joblib" in model_path:    
        scaler_path = "".join(model_path.split(".joblib")) + model_name + "_scaler.joblib"
    elif ".pkl" in model_path:
        scaler_path = "".join(model_path.split(".pkl")) + model_name + "_scaler.joblib"
    else:
        scaler_path = os.path.join(model_path,f"{model_name}_scaler.joblib")
    #scaler_path = "./box/models/defect_classify/model_scaler.joblib"
    
    if f"{model_name}.joblib" not in model_path:
        model_path = os.path.join(model_path,"model.joblib")
        
    # load robust scaler
    with open(scaler_path, "rb") as fh:
        robust_scaler = joblib.load(fh.name)
    
    # load daal model
    with open(model_path, "rb") as fh:
        daal_model = joblib.load(fh.name)
    categorical_columns = []
    
    data = pd.DataFrame(data, index=[0])
    scaled_samples_transformed = robust_scaler.transform(data)

    if len(categorical_columns)>0:
        processed_sample = pd.concat([scaled_samples_transformed, data], axis=1)
    else:
        processed_sample = scaled_samples_transformed
    
    daal_predict_algo = d4p.gbt_classification_prediction(
            nClasses=num_class,
            resultsToEvaluate="computeClassLabels",
            fptype='float')
    daal_prediction = daal_predict_algo.compute(processed_sample, daal_model)
    print(daal_prediction.prediction[:, 0])
    status = 'unknown defect'
    for prediction in daal_prediction.prediction[:, 0]:
        if prediction == 0:
            status = 'No Defect'
            return status
        elif prediction == 1:
            status = 'Type-1 Defect'
            return status
        elif prediction == 2:
            status = 'Type-2 Defect'
    return status

