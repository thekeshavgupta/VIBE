import torch
from torch import nn
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from Adversary import Adversary
from Encoder import Encoder

class MainModel():
    def __init__(self, dataPath: str):
        self.encoder = Encoder(dataPath)
        [self.input_data, self.output_data] = self.encoder.encode()
        
        self.loss_fnc = nn.BCELoss()
        self.advModel = Adversary(self.input_data.shape[1])
        self.optimiser = torch.optim.Adam(self.advModel.parameters(), lr=0.01)
        self.num_epochs = 1000
    
    def trainModel(self):
        input_train, input_test, output_train, output_test = train_test_split(self.input_data, self.output_data, test_size=0.3)
        for epochs in range(self.num_epochs):
            self.advModel.train()
            self.optimiser.zero_grad()
            outputs = self.advModel(input_train)
            loss_value = self.loss_fnc(outputs, output_train)
            loss_value.backward()
            self.optimiser.step()
            print(f"==> Epoch: {epochs} Training Loss: {str(loss_value)}") 
        
        self.advModel.eval()
        with torch.no_grad():
            test_pred = self.advModel(input_test)
            test_loss = self.loss_fnc(torch.round(test_pred), output_test)
            print(f"Test Loss: {test_loss}")           
    
    def run(self):
        print("==== Started the execution ==== ")
        self.trainModel()
        print("==== Execution completed ==== ")
    
    
if __name__ == "__main__":
    model_orchestrator = MainModel('biased_gender_data_synthetic.csv')
    model_orchestrator.run()
    