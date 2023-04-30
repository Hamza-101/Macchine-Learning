import torch
import random
import statistics
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ABBREVIATIONS
# 1) BCE -> Binary Cross Entropy error
# 2) SD -> Standard Deviation

#DESIGN PARAMETERS
k = 7                               #Neighbors in KNN
numOfEpochs = 50                    #Epochs
lambdaval=0.001
training_loss=[]                    #Array to store loss of an epoch
TotalEpochs=[]                      #Array for the Number of Epoch as members
PATH = "MSCS22034_Model.pt"         #Model name
learningrate=0.001
output={"training_loss":[], "bce_mean": 0.0, "bce_sd": 0.0}   #Dictionary to store outputs


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

#Changing value of -1 to 0 to refit our data for sigmoid activation unction
Ytr[Ytr == (-1.0)] = 0.0

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

def trainingOnDataset():
    epochs_total=numOfEpochs
    cycles=[]
    bceloss=[]
    epoch=1
  
    #Getting random epoch values for sampling
    cycles.append(random.randint(0,10))
    cycles.append(random.randint(10,20))
    cycles.append(random.randint(20,30))
    cycles.append(random.randint(30,40))
    cycles.append(random.randint(40,50))

    trainingLossPlot=[]

    neural_network_instance.train()
    while (epochs_total != 0):
        current_loss = 0.0
        l1Loss = 0.0
        for data in train_dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = neural_network_instance(inputs)
            bce_loss = nn.BCELoss()(outputs, labels)
            l1Loss = lambdaval * torch.norm(outputs, p=1)
            loss = bce_loss + l1Loss
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

        if(epoch in cycles): 
            bceloss.append(float(loss))
       
        training_loss = current_loss / len(train_dataloader)
        trainingLossPlot.append(float(round(training_loss, 4)))

       
        print("----------------")
        print("\n")
        print(f"Epoch: {epoch}")
        print(f"Training Loss = {round(training_loss, 4)}")
        print("\n")
        print("----------------")
        epoch+=1
        epochs_total = epochs_total - 1


    bce_mean=statistics.mean(bceloss)
    bce_std=statistics.stdev(bceloss)

    print('Model Trained.')
    
    return trainingLossPlot, bce_mean, bce_std

def loss_variation(loss,epochs):
    plt.xlabel('Number of Epoch')
    plt.ylabel('Training Loss')
    plt.title('Loss Variation')

    plt.plot(epochs, loss)
    plt.show()


neural_network_instance = Neural_Network().to(device)
optimizer = torch.optim.SGD(neural_network_instance.parameters(), lr=learningrate)
data_train = dataTrain(X_train, Y_train)
train_dataloader = DataLoader(data_train, batch_size=196, shuffle=True)

#Returning output from trained model 
output=trainingOnDataset()

#Saving trained model
torch.save(neural_network_instance.state_dict(), PATH)

#STATISTICAL RESULTS
print("\n")
print("BCE Mean: ", (round(output[1], 4)))
print("\n")
print("BCE SD: ", (round(output[2], 4)))
print("\n")


#Plotting the learning curve
loss_variation(output[0], TotalEpochs)

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
