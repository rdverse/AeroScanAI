from pydantic import BaseModel

class PredictionPayload(BaseModel):
    data: list
    model_path: str
    num_class: int = 3
    scaler: bool = True
    
class TrainPayload(BaseModel):
    file: str
    model_name: str
    model_path: str
    test_size: int = 25  
    ncpu: int = 4 
    
class AppendDataPayload(BaseModel): 
    data_path: str
    data: list
    