import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

class Encoder():
    def __init__(self, dataPath: str):
        self.data = pd.read_csv(dataPath, sep=',')
    
    def encode(self, input_column: str = '', output_column: str = ''):
        print("**** Generating the Embeddings of the input data ****")
        embeddingModel = SentenceTransformer('bert-base-nli-mean-tokens')
        input_data = torch.tensor(embeddingModel.encode(self.data[input_column].tolist()))
        output_data = torch.tensor(self.data[output_column].tolist(), dtype=torch.float32).reshape(-1,1)
        print("**** Embeddings generation completed ****")
        return [input_data, output_data]
        