import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Adversary import Adversary
from Encoder import Encoder

class VIBE():
    def __init__(self,input_col: str, output_col: str, dataPath: str, hidden_layer_sizes: list[int], learning_rate: float = 0.001, epochs: int = 500, enable_debiasing = False):
        self.input_col = input_col
        self.output_col = output_col
        self.encoder = Encoder(dataPath)
        self.learning_rate = learning_rate
        self.loss_fnc = nn.BCELoss()
        self.num_epochs = epochs
        self.hidden_layers = hidden_layer_sizes
        self.enable_debiasing = enable_debiasing
        if(len(self.hidden_layers) == 0):
            raise SyntaxError("hidden layers cannot be empty")
    
    def trainModel(self):
        # Generating the embeddings of the inputs
        [self.input_data, self.output_data] = self.encoder.encode(self.input_col, self.output_col)
        self.advModel = Adversary(self.input_data.shape[1], self.hidden_layers, self.enable_debiasing)
        self.optimiser = torch.optim.Adam(self.advModel.parameters(), lr=self.learning_rate)
        
        input_train, input_test, output_train, output_test = train_test_split(self.input_data, self.output_data, test_size=0.3)
        for epochs in range(self.num_epochs):
            self.advModel.train()
            self.optimiser.zero_grad()
            outputs = self.advModel(input_train)
            loss_value = self.loss_fnc(outputs, output_train)
            loss_value.backward()
            self.optimiser.step()
            if(self.enable_debiasing):
                input_train = self.advModel.getImportantFeatureRepresentations(input_train)
            print(f" ==> Epoch: {epochs} Training Loss: {str(loss_value)}") 
        
        self.advModel.eval()
        with torch.no_grad():
            test_pred = self.advModel(input_test)
            test_loss = self.loss_fnc(test_pred, output_test)
            print(f"*** Test Loss: {test_loss} ***")
            test_pred_labels = (test_pred > 0.5).float()
            test_accuracy = accuracy_score(output_test.cpu(), test_pred_labels.cpu())
            print(f"*** Test Accuracy: {test_accuracy} ***")
            print("**** Model training completed ****")
    def fetch_debiased_embeddings(self):
        return self.advModel.getImportantFeatureRepresentations(self.input_data)
    
    def run(self):
        print("==== Started the execution ==== ")
        self.trainModel()
        print("==== Execution completed ==== ")

    