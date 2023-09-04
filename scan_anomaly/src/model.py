from pydantic import BaseModel

class PredictionPayload(BaseModel):
    model_path: str
    data_path: str
    num_class: int = 2
    # remove scaler since random forest won't have much influence of scaling features
    
class TrainPayload(BaseModel):
    n_samples: int
    img_dim: int
    defect_coverage: float
    file: str
    model_name: str
    model_path: str
    ncpu: int = 1
    