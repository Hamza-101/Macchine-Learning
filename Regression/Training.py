import torch
import random
import statistics
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ABBREVIATIONS
# 1) RMSE -> Root mean square error
# 2) SD -> Standard Deviation

#DESIGN PARAMETERS
numOfEpochs = 200                   #Epochs
l1Reg_lambda=0.001                  #L1 Regulaization lambda value
training_loss=[]                    #Array to store loss for each epoch
TotalEpochs=[]                      #Array for the Number of Epoch as members
PATH = "MSCS22034_Model.pt"         #Model name
learningrate=0.0001
output={"trainingLoss":[], "rmse_mean": 0.0, "rmse_sd": 0.0}   #Dictionary to store outputs


#Initializing the array TotalEpochs with epoch number for the plot
TotalEpochs = [i for i in range(numOfEpochs)] 


#Setting up parameter to use gpu 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


#Reading features and labels files as data frames
Xtr = pd.read_csv("TrainData.csv", header=None, sep=" ")  
Ytr = pd.read_csv("TrainLabels.csv",header=None, sep=" ")

#Converting our data frames to arrays
Xtr=Xtr.to_numpy()
Ytr=Ytr.to_numpy()

#Setting training parameters
X_train = Xtr.astype(np.float32)  
Y_train = Ytr.astype(np.float32) 

#Data formatting to tensor
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

def RMSELoss(yHat, y):
    mse = torch.mean((yHat - y) ** 2)
    l1_reg = torch.tensor(0.).to(device)
    
    for name, param in neural_network_instance.named_parameters():
        if 'weight' in name:
            l1_reg += torch.norm(param, 1)
    error = torch.sqrt(mse + l1Reg_lambda * l1_reg)

    return error
def trainingOnDataset():
    epochs_total=numOfEpochs
    cycles=[]
    rmse=[]
    epoch=1

    #Getting random epoch values for sampling
    cycles.append(random.randint(0,40))
    cycles.append(random.randint(40,80))
    cycles.append(random.randint(80,120))
    cycles.append(random.randint(120,160))
    cycles.append(random.randint(160,200))

    trainingLossPlot=[]

    while (epochs_total != 0):
        current_loss = 0.0

        for data in train_dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = neural_network_instance(inputs)
            loss = RMSELoss(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            current_loss += loss.item()   

        if(epochs_total in cycles):    
            rmse.append(float(loss))

        training_loss = (current_loss / len(train_dataloader))
        trainingLossPlot.append(float(round(training_loss, 4)))

        

        print("----------------")
        print("\n")
        print(f"Epoch: {epoch}")
        print(f"Training Loss = {round(training_loss, 4)}")
        print("\n")
        print("----------------")
        epoch+=1
        epochs_total = epochs_total - 1
        
       
   
    print('Model Trained')

    rmse_mean=statistics.mean(rmse)
    rmse_std=statistics.stdev(rmse)
   
    return trainingLossPlot, rmse_mean, rmse_std
def loss_variation(loss,epochs):
    plt.xlabel('Number of Epoch')
    plt.ylabel('Training Loss')
    plt.title('Loss Variation')

    plt.plot(epochs, loss)
    plt.show()

neural_network_instance=Neural_Network().to(device)

#OPTIMIZER - Stochastic Gradient Descent
optimizer = torch.optim.SGD(neural_network_instance.parameters(), lr=learningrate, weight_decay=l1Reg_lambda)
 
#Data Loader initialization
data_train = dataTrain(X_train, Y_train)
train_dataloader = DataLoader(data_train, batch_size=196, shuffle=True) #Shuffle option makes the data be trained patternless



#Returning output from trained model 
output=trainingOnDataset()

#Saving trained model
torch.save(neural_network_instance.state_dict(), PATH)


#STATISTICAL RESULTS
print("\n")
print("RMSE Mean: ", (round(output[1], 4)))
print("\n")
print("RMSE SD: ", (round(output[2], 4)))
print("\n")



#Plotting the learning curve
loss_variation(output[0], TotalEpochs)
