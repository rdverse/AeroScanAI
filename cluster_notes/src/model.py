from pydantic import BaseModel


class ClusterPayload(BaseModel):
    data_name: str = 'cluster'
    data_path: str = './box/datasets/waveform_probe/'
    