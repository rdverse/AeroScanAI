import uvicorn
from fastapi import FastAPI
import logging
import warnings
from predict import inference 
import pandas as pd
import inspect
from data_utils import append_data, fetch_coordinates_data
from fastapi import FastAPI
from model import TrainPayload, PredictionPayload, AppendDataPayload, FetchCoordinatesPayload
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
    print(sample)
    results = inference(model_name = payload.model_name, model_path = payload.model_path, data=sample, num_class = payload.num_class)
    return {"msg": "Completed Analysis", "Defect Result": results}

@app.post("/fetch_coordinates_data")
async def fetch_data(payload:FetchCoordinatesPayload):
    print("entered fetch coordinates data")
    fetched_data = fetch_coordinates_data(img_dim=payload.img_dim, n_channels=payload.n_channels, test_scan = payload.test_scan, x_coord = payload.x_coord, y_coord = payload.y_coord)
    print("append_data args")
    return {"msg": "Completed fetching coordinates data", "fetched_data is": fetched_data, "fetched_data": fetched_data}

@app.post("/append_data")
async def new_data_append(payload:FetchCoordinatesPayload):
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