import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

class Encoder():
    def __init__(self, dataPath: str):
        self.data = pd.read_csv(dataPath, sep=',')
    
    def encode(self):
        embeddingModel = SentenceTransformer('bert-base-nli-mean-tokens')
        input_data = torch.tensor(embeddingModel.encode(self.data['bio'].tolist()))
        output_data = torch.tensor(self.data['gender'].tolist(), dtype=torch.float32).reshape(-1,1)
        return [input_data, output_data]
    
    def encodeOnTheFly(self, input: torch.tensor, output: torch.tensor):
        pass
        