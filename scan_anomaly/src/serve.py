import uvicorn
from fastapi import FastAPI
import logging
import warnings
import pandas as pd
import inspect
import numpy as np
from fastapi import FastAPI
from model import TrainPayload, PredictionPayload
from torchipex.training.TrainModel import TrainModel
from torchipex.unet.unet_model import UNet#, DoubleConv, Down, Up, OutConv, FullyConected, GlobalAvgPool
from sklearn.preprocessing import MinMaxScaler
app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.get("/ping")
async def ping():
    """Ping server to determine status

    Returns
    -------
    API response
        response from server on health status
    """
    return {"message":"Server is Running"}

@app.post("/train")
async def train(payload:TrainPayload):
    """Training Endpoint
    This endpoint process raw data and trains an XGBoost Classifier and converts it to daal4py format.

    Parameters
    ----------
    payload : TrainPayload
        Training endpoint payload model
    Returns
    -------
    API response
        Accuracy metrics and other logger feedback on training progress.
    """
    unetModel = TrainModel() 
    unetModel.load_model(n_channels=payload.n_channels, 
                                 n_classes=payload.n_classes, 
                                 img_dim = payload.img_dim)
    unetModel.load_data(n_samples=payload.n_samples,
                        percent_test=payload.percent_test,
                        batch_size=payload.batch_size,
                        num_workers=payload.n_cpus) 
    unetModel.train(n_epochs=payload.n_epochs) 
    
    unetModel.save_model(model_name= payload.model_name,model_path=payload.model_path)
    
    eval_df = unetModel.evaluate()
    print(eval_df)
    return_dict = {"msg": "Completed Training",
                     "results": eval_df.to_dict()}
        
    return return_dict

@app.post("/predict")
async def predict(payload:PredictionPayload):
    """Prediction Endpoint
    This endpoint uses a trained model to make predictions on test data.

    Parameters
    ----------
    payload : PredictionPayload
        Prediction endpoint payload model

    Returns
    -------
    dict
        Predicted values, labels, and input data.
    """
    unetModel = TrainModel() 
    unetModel.load_model(n_channels=payload.n_channels, 
                                 n_classes=payload.n_classes, 
                                 img_dim = payload.img_dim)
    unetModel.load_model_from_file(model_name= payload.model_name,
                                   model_path=payload.model_path)
    test_scan_data = unetModel.load_single_scan(payload.test_scan)
    print(test_scan_data)
    preds, labels, inputs = unetModel.predict(test_scan_data)
    print(preds.shape)
    print(labels.shape)
    print(inputs.shape)
    inputs = np.std(inputs.squeeze(), axis=0) 
    scale = MinMaxScaler()
    inputs = scale.fit_transform(inputs.reshape(-1, 1)).reshape(payload.img_dim, payload.img_dim)*255
    inputs = str(inputs.astype(np.int32).tolist())
    preds = str(preds.astype(np.int32).tolist())
    labels = str(labels.astype(np.int32).tolist())
    return_dict = {"msg": "Completed Prediction",
                     "preds": preds,
                     "labels": labels,
                     "inputs": inputs}
    return return_dict
    
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=5003, log_level="info")
