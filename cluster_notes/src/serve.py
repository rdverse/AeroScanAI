import uvicorn
from fastapi import FastAPI
import logging
import warnings
import pandas as pd
import numpy as np
from fastapi import FastAPI
from sklearn.preprocessing import MinMaxScaler
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

@app.post("/cluster")
async def train(payload:TrainPayload):
   
    # test one - being able to load the model properly
    unetModel = TrainModel() 
    unetModel.load_model(n_channels=payload.n_channels, 
                                 n_classes=payload.n_classes, 
                                 img_dim = payload.img_dim)
    #print(unetModel.model)
    #print(unetModel.__dict__)
    # test two - being able to load the data properly
    unetModel.load_data(n_samples=payload.n_samples,
                        percent_test=payload.percent_test,
                        batch_size=payload.batch_size,
                        num_workers=payload.n_cpus) 
    # test 3 run some supervised training
    unetModel.train(n_epochs=payload.n_epochs) 
    
    unetModel.save_model(model_name= payload.model_name,model_path=payload.model_path)
    
    eval_df = unetModel.evaluate()
    print(eval_df)
    return_dict = {"msg": "Completed Training",
                     "results": eval_df.to_dict()}
        
    return return_dict

@app.post("/predict")
async def predict(payload:PredictionPayload):
    #sample = pd.json_normalize(payload.data) 
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
    
    #unetModel.predict(test_scan_data)
    #unetModel.load_model(n_channels=payload.n_channels,
 
#add implementation for retraining the model 
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=5004, log_level="info")