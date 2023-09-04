import uvicorn
from fastapi import FastAPI
import logging
import warnings
from predict import WaveInference
import pandas as pd
import inspect
from fastapi import FastAPI
from model import TrainPayload, PredictionPayload
from train import WaveformProbe

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
    model = WaveformProbe(payload.model_name)
    # loads data
    model.process_data(payload.file, payload.test_size)
    logger.info("Data has been successfully processed")
    fi = model.train(payload.ncpu)
    logger.info("Robotic Maintenance Model Successfully Trained")
    model.save(payload.model_path)
    logger.info("Saved Robotic Maintenance Model")
    accuracy_score = model.validate()
    return {"msg" : f"{fi} Model trained succesfully model path is {payload.model_path}\n feature importance : {fi}",
            "validation scores": accuracy_score}

@app.post("/predict")
async def predict(payload:PredictionPayload):
    #sample = pd.json_normalize(payload.data) 
    inferenceEngine = WaveInference(payload.model_path, payload.data_path)
    inferenceEngine.load_model_and_data()
    results, wavetoplot = inferenceEngine.inference(x_coord=payload.x_coord , y_coord=payload.y_coord)
    feature_importance = inferenceEngine.lime_explanation(x_coord=payload.x_coord , y_coord=payload.y_coord)
    print("back to predict")
    print(results)
    print(wavetoplot)
    return {"msg": "Completed Analysis", "results": results, "wavetoplot": str(list(wavetoplot[0])), "feature_importance": str(list(feature_importance))}

# @app.post("/plotwaveform")
# async def plotwaveform(payload:PredictionPayload):
#     print("entered plotwaveform")
#     #sample = pd.json_normalize(payload.data)
#     data = get_data(x_coord=payload.x_coord , y_coord=payload.y_coord)    
#     return {"msg": "Fetched waveform", "waveform": data}


#add implementation for retraining the model 
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=5002, log_level="info")