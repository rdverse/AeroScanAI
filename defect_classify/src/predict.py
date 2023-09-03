import daal4py as d4p
import joblib
import os
# import pdb
import numpy as np
import pandas as pd

def inference(model_path, data, num_class: int = 3, scaler: bool = True):
    # Use .joblib extension to save file
    scaler_path = ""
    if "model" in model_path:
        scaler_path = scaler_path.replace("model", "")
        
    if ".pb" in model_path:
        scaler_path = "".join(model_path.split(".pb")) + "model_scaler.joblib"
    elif ".joblib" in model_path:    
        scaler_path = "".join(model_path.split(".joblib")) + "model_scaler.joblib"
    elif ".pkl" in model_path:
        scaler_path = "".join(model_path.split(".pkl")) + "model_scaler.joblib"
    else:
        scaler_path = os.path.join(model_path,"model_scaler.joblib")
    #scaler_path = "./box/models/defect_classify/model_scaler.joblib"
    
    if "model.joblib" not in model_path:
        model_path = os.path.join(model_path,"model.joblib")
        
    # load robust scaler
    with open(scaler_path, "rb") as fh:
        robust_scaler = joblib.load(fh.name)
    
    # load daal model
    with open(model_path, "rb") as fh:
        daal_model = joblib.load(fh.name)
    categorical_columns = []
    #return str(os.listdir("./box/defect_classify")) + "   " + str(os.listdir()) + "   " + model_path + "   "  + scaler_path #os.listdir(sca)))    
    # process data sample
    # Categorical_Variables = pd.get_dummies(
    #                        data[[
    #                            'Manufacturer',
    #                            'Generation',
    #                            'Lubrication',
    #                            'Product_Assignment']],
    #                        drop_first=False)
    # data = pd.concat([data, Categorical_Variables], axis=1)
    # data.drop(['Manufacturer', 'Generation', 'Lubrication', 'Product_Assignment'], axis=1, inplace=True)
    # data = data.astype({'Motor_Current': 'float64', 'Number_Repairs': 'float64'})
    print(data)
    #number_samples = data.select_dtypes(['float', 'int', 'int32'])
    data = pd.DataFrame(data, index=[0])
    #np.array(list(data.values())).reshape(1, -1)
    print(data)
    scaled_samples_transformed = robust_scaler.transform(data)
    # scaled_samples_transformed = pd.DataFrame(scaled_samples,
    #                                              index=number_samples.index,
    #                                              columns=number_samples.columns)
    # del scaled_samples_transformed['Number_Repairs']
    # data = data.drop(['Age', 'Temperature', 'Last_Maintenance', 'Motor_Current'], axis=1)
    #data = data.astype(int)
    if len(categorical_columns)>0:
        processed_sample = pd.concat([scaled_samples_transformed, data], axis=1)
    else:
        processed_sample = scaled_samples_transformed
    # processed_sample = processed_sample.astype({'Motor_Current': 'float64'})
        
    # column_names = ['Age', 'Temperature', 'Last_Maintenance', 'Motor_Current',
    #             'Number_Repairs', 'Manufacturer_A', 'Manufacturer_B',
    #             'Manufacturer_C', 'Manufacturer_D', 'Manufacturer_E', 'Manufacturer_F',
    #             'Manufacturer_G', 'Manufacturer_H', 'Manufacturer_I', 'Manufacturer_J',
    #             'Generation_Gen1', 'Generation_Gen2', 'Generation_Gen3',
    #             'Generation_Gen4', 'Lubrication_LTA', 'Lubrication_LTB',
    #             'Lubrication_LTC', 'Product_Assignment_PillA',
    #             'Product_Assignment_PillB', 'Product_Assignment_PillC']
    # column_names = ['backwall', 'frontwall', 'ramp', 'geometry', 'no_peaks', 'noise', 'max', 'min', 'signal_noise_ratio']
    # zeroes_dataframe = pd.DataFrame(0, index=np.arange(1), columns=column_names)
    # merged_df = pd.merge(zeroes_dataframe, processed_sample, on=processed_sample.columns.tolist(), how='right').fillna(0)
    # perform inference
    daal_predict_algo = d4p.gbt_classification_prediction(
            nClasses=num_class,
            resultsToEvaluate="computeClassLabels",
            fptype='float')
    daal_prediction = daal_predict_algo.compute(processed_sample, daal_model)
    print(daal_prediction.prediction[:, 0])
    status = 'unknown defect'
    for prediction in daal_prediction.prediction[:, 0]:
        if prediction == 0:
            status = 'Equipment Does Not Require Scheduled Maintenance'
            return status
        elif prediction == 1:
            status = 'Equipment Requires Scheduled Maintenance - Plan Accordingly'
            return status
        elif prediction == 2:
            status = 'Equipment Requires Immediate Maintenance - Schedule Immediately'
    return status


def append_data(data_path, data):
    """
    Append new data to an existing or new pickle file.

    Parameters
    ----------
    data_path : str
        The file path to the pickle file where data will be stored or appended.
    data : pd.DataFrame
        The DataFrame containing the new data to be appended.

    Returns
    -------
    int
        The total number of records in the combined DataFrame after appending.

    Notes
    -----
    If the specified `data_path` does not exist, a new pickle file will be created.
    If the file already exists, the new data will be appended to the existing data.
    """
    combined_data = pd.DataFrame()
    previousdatalen = 0
    # Check if the data file already exists
    if os.path.exists(data_path):
        # Read the existing pickle file into a DataFrame
        existing_data = pd.read_pickle(data_path)
        previousdatalen = len(existing_data)
    else:
        existing_data = None

    if existing_data is not None:
        # Append the new data to the existing data
        combined_data = pd.concat([existing_data, data], ignore_index=True)
        # Save the combined data back to the pickle file
        combined_data.to_pickle(data_path)
        print("Data appended to existing pickle file.")
    else:
        # If no existing data found, create a new pickle file
        data.to_pickle(data_path)
        print("New data file created.")

    print("Data saved to pickle file.")
    return {"msg": f"Data points before appending : {previousdatalen}, and data points after appending: {len(combined_data)}"}
