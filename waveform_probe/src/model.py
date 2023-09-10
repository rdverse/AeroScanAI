from pydantic import BaseModel

class PredictionPayload(BaseModel):
    # x_coord: int
    # y_coord: int
    # model_path: str
    # data_path: str
    # num_class: int = 2
    img_dim : int
    n_channels : int
    test_scan : str
    model_name : str
    model_path: str
    x_coord: int
    y_coord: int
    num_class: int = 2
    scaler: bool = False    
    # remove scaler since random forest won't have much influence of scaling features
    
class TrainPayload(BaseModel):
    img_dim: int
    n_channels: int
    test_scan: str 
    train_scan: str 
    model_name: str
    model_path: str
    append_path: str
    ncpu: int = 1
    
class FetchCoordinatesPayload(BaseModel):
    img_dim : int
    n_channels : int
    test_scan : str
    x_coord : int
    y_coord : int
    
