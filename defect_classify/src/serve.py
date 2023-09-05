import uvicorn
from fastapi import FastAPI
import logging
import warnings
from predict import inference, append_data
import pandas as pd
import inspect
from fastapi import FastAPI
from model import TrainPayload, PredictionPayload, AppendDataPayload
from train import DefectClassify

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#warnings.filterwarnings("ignore")

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
    model = DefectClassify()
    model.process_data(payload.img_dim, payload.n_channels, payload.train_scan, payload.test_scan, payload.append_path)
    print("Data has been successfully processed")
    logger.info("Data has been successfully processed")
    fi = model.train(payload.ncpu)
    logger.info("XGBoost Model Successfully Trained")
    mp, sp = model.save(payload.model_path, payload.model_name)
    logger.info("Saved XGBoost Model Successfully")
    accuracy_score = model.validate()
    return {"msg" : f"{mp}   {sp} Model trained succesfully model path is {payload.model_path}\n feature importance : {fi}", "validation scores": accuracy_score}

@app.post("/predict")
async def predict(payload:PredictionPayload):
    print("entered predict")
    sample = pd.json_normalize(payload.data)
    results = inference(data = sample, model_path = payload.model_path, num_class = payload.num_class)
    return {"msg": "Completed Analysis", "Maintenance Recommendation": results}

@app.post("/append_data")
async def new_data_append(payload:AppendDataPayload):
    sample = pd.json_normalize(payload.data)
    print("sample in append data")
    print(sample)
    print("append_data args")
    print(inspect.signature(append_data).parameters)
    print(append_data.__dict__)
    results = append_data(data_path = payload.data_path, data = sample)
    #results = inference(data = sample, model_path = payload.model)
    return {"msg": "Completed appending data", "length of data now is": results}

#add implementation for retraining the model 
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=5001, log_level="info")