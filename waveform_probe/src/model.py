from pydantic import BaseModel

class PredictionPayload(BaseModel):
    x_coord: int
    y_coord: int
    model_path: str
    data_path: str
    num_class: int = 2
    # remove scaler since random forest won't have much influence of scaling features
    
class TrainPayload(BaseModel):
    file: str
    model_name: str
    model_path: str
    test_size: int = 25
    ncpu: int = 1
    
