import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader


# Setup Environment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#Loading model
filename = "MSCS22034_Model.pt"
model = torch.load(filename)

#Reading features and labels files as data frames
Xtr = pd.read_csv("TrainData.csv", header=None, sep=" ")  
Ytr = pd.read_csv("TrainLabels.csv",header=None, sep=" ")

#Converting our data frames to arrays
Xtr=Xtr.to_numpy()
Ytr=Ytr.to_numpy()

#Setting training parameters
X_train = Xtr.astype(np.float32)  
Y_train = Ytr.astype(np.float32) 

class dataTrain(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(device), self.Y[idx].to(device)

class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(Xtr.shape[1], 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layers(x)

def predictionNN(model, testData):
    model.eval()                            #inbuilt function
    predictions = []
    
    with torch.no_grad():
        for inputs, _ in testData:
            inputs = inputs.to(device)      #Converting to tensor
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())      #Converting to and appending a numpy array

    return predictions
 
data_train = dataTrain(X_train, Y_train)
testing_dataloader = DataLoader(data_train, batch_size=196, shuffle=False)
neural_network_instance = Neural_Network().to(device)
neural_network_instance.load_state_dict(model)
predictionsNN=predictionNN(neural_network_instance,testing_dataloader)

print("Predictions: ", predictionsNN)
print("\n")
print("\n")
print("\n")
pd.DataFrame(predictionsNN).to_csv("MSCS22034_Predictions.csv")
print("Predictions saved")
