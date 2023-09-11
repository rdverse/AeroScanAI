import os
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
warnings.filterwarnings("ignore")

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
    model.process_data(payload.img_dim, payload.n_channels, payload.train_scan, payload.test_scan, payload.append_path)
    logger.info("Data has been successfully processed")
    fi = model.train(payload.ncpu)
    logger.info("Robotic Maintenance Model Successfully Trained")
    model.save(payload.model_path)
    logger.info("Saved Robotic Maintenance Model")
    benchmarks = model.validate()
    print(benchmarks)
    return {"msg" : f"Model trained succesfully model path is {os.path.join(payload.model_path, payload.model_name)}",
            "benchmarks": benchmarks}
    
    
@app.post("/predict")
async def predict(payload:PredictionPayload):
    #sample = pd.json_normalize(payload.data) 
    inferenceEngine = WaveInference(model_name=payload.model_name, model_path=payload.model_path)
    
    inferenceEngine.load_model_and_data(n_channels=payload.n_channels, 
                                        img_dim=payload.img_dim,
                                        test_scan=payload.test_scan)
    
    results, wavetoplot = inferenceEngine.inference(x_coord=payload.x_coord , y_coord=payload.y_coord)
    feature_importance = inferenceEngine.lime_explanation()
    print("back to predict")
    print(results)
    print(wavetoplot)
    print("serve funtino inference ")
    
    return_dict = {"msg": "Completed Analysis", 
                     "results": results, 
                     "wavetoplot": str(list(wavetoplot[0])),
                     "feature_importance": str(list(feature_importance))}
    print(return_dict)
    return return_dict

    #return{"sampleresponse": "sample response"}
# @app.post("/plotwaveform")
# async def plotwaveform(payload:PredictionPayload):
#     print("entered plotwaveform")
#     #sample = pd.json_normalize(payload.data)
#     data = get_data(x_coord=payload.x_coord , y_coord=payload.y_coord)    
#     return {"msg": "Fetched waveform", "waveform": data}


#add implementation for retraining the model 
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=5002, log_level="info")