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
X_test = pd.read_csv("TestLabels.csv",header=None, sep=" ")
Y_test = pd.read_csv("TestLabels.csv",header=None, sep=" ")


#Converting our data frames to arrays
Xtr=Xtr.to_numpy()
Ytr=Ytr.to_numpy()
X_test=X_test.to_numpy()
Y_test=Y_test.to_numpy()

#Setting training parameters
X_train = Xtr.astype(np.float32)  
Y_train = Ytr.astype(np.float32) 
X_test = X_test.astype(np.float32)
Y_test = Y_test.astype(np.float32)

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
            nn.Linear(Xtr.shape[1], 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
def predictionNN(model, testData):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in testData:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs[outputs>=0.5]=1
            outputs[outputs<0.5]=0
            predictions.extend(outputs.cpu().numpy())
    return predictions



#######################################
#######################################
#                 KNN                 #
#######################################
#######################################

def KNN(x_test, X_train,y_train, k):
    distances=[]
    for i, x_train in enumerate(X_train):
        dist = np.sqrt(np.sum((x_test - x_train) ** 2))
        distances.append((dist, y_train[i]))
    distances = sorted(distances, key=lambda x: x[0])
    k_neighbors = distances[:k]
    k_labels = []
    for i in range(len(k_neighbors)):
        k_labels.append(tuple(k_neighbors[i][1].tolist()))

    most_common = max(set(k_labels), key=k_labels.count)
    return most_common

def predictionKNN(testdataset):
    predictions=[]

    for x_test in testdataset:
        predictions.extend(KNN(x_test, X_test,Y_test, 3))
        
    predictions = np.array(predictions)
    return predictions

filename = "MSCS22034_Model.pt"
model = torch.load(filename)
 
data_train = dataTrain(X_train, Y_train)
testing_dataloader = DataLoader(data_train, batch_size=196, shuffle=False)
neural_network_instance = Neural_Network().to(device)
neural_network_instance.load_state_dict(model)

print("-------------------------------------")
print("                 NN                 ")
print("-------------------------------------")
predictionsNN = predictionNN(neural_network_instance, testing_dataloader)
print(predictionsNN)

print("-------------------------------------")
print("                 KNN                 ")
print("-------------------------------------")
predictionsKNN=predictionKNN(X_test)
print(predictionsKNN)



pd.DataFrame(predictionsNN).to_csv("MSCS22034_Predictions.csv")
pd.DataFrame(predictionsKNN).to_csv("MSCS22034_Predictions_KNN.csv")
print("Predictions saved")  

