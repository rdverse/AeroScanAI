from pydantic import BaseModel

class PredictionPayload(BaseModel):
    data: list
    model_name: str
    model_path: str
    num_class: int = 3
    scaler: bool = True
    
class TrainPayload(BaseModel):
    img_dim: int
    n_channels: int
    test_scan: str 
    train_scan: str 
    model_name: str
    model_path: str
    append_path: str
    ncpu: int = 1
    
class AppendDataPayload(BaseModel): 
    data_path: str
    data: list
    
class FetchCoordinatesPayload(BaseModel):
    img_dim : int
    n_channels : int
    test_scan : str
    x_coord : int
    y_coord : int