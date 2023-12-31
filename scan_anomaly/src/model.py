from pydantic import BaseModel

class PredictionPayload(BaseModel):
    # active learning does not apply here, only active inference
    al_threshold: float = 0.5
    model_name : str
    model_path: str
    img_dim : int
    n_channels : int
    n_classes : int = 1
    test_scan : str
    num_class: int = 1 
    
class TrainPayload(BaseModel):
    active_learning: bool = False
    al_threshold: float = 0.5
    img_dim: int
    n_channels: int
    n_classes: int = 1
    n_samples: int
    model_name: str
    model_path: str
    n_cpus: int = 8
    n_epochs: int = 10
    batch_size: int = 32
    percent_test: float = 0.2
    