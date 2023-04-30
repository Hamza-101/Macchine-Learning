import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import statistics
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm



output={"trainingLoss":[], "rmse_mean": 0.0, "rmse_sd": 0.0} 
torch.backends.cudnn.enabled = False

# train = scipy.io.loadmat('train_data.mat')
train_data = scipy.io.loadmat('train_data.mat')
train_data=train_data["train_data"]

test_data = scipy.io.loadmat('test_data.mat')


train_data=torch.tensor(train_data)


# #labels
label_train_data = scipy.io.loadmat('./lists/train_list.mat')
label_train_data=label_train_data["labels"]
label_train_data=torch.tensor(label_train_data)





learningrate=0.1

criterion = nn.CrossEntropyLoss()
# Define convolutional neural network
class Net(nn.Module):
    def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv1d(1, 22, kernel_size=3)
            self.conv2 = nn.Conv1d(22, 20, kernel_size=3)
            self.conv3 = nn.Conv1d(20, 3, kernel_size=3)
            self.conv4 = nn.Conv1d(3, 10, kernel_size=3)
            self.flatten = nn.Flatten()  # add a flatten layer
            
            self.fc1 = nn.Linear(10 *11992, 20)
            self.fc2 = nn.Linear(20, 36)
            self.fc3 = nn.Linear(36,120)
            self.activation=nn.Sigmoid()

    def forward(self, x):
            x  = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.relu(self.conv3(x))
            x = nn.functional.relu(self.conv4(x))
            x = self.flatten(x)  # add a flatten layer
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x


# Define a function for training the model
def train():

    epochs_total=0
    rmse=[]
    cycles=[]
    cycles.append(random.randint(0,10))
    cycles.append(random.randint(10,20))

    cycles.append(random.randint(20,30))
    cycles.append(random.randint(30,40))
    cycles.append(random.randint(40,50))
    trainingLossPlot=[]    
    
    
    for epoch in tqdm(range(100)):
        
        running_loss = 0.0
        for i in tqdm(train_dataloader):
            inputs,labels=i
             
        optimizer.zero_grad()
          
        outputs = model(inputs)

        loss= criterion(outputs,labels.squeeze())
        
        print("LOSS=", loss)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
           
        if(epoch in cycles):    
            rmse.append(float(loss))
        print('Epoch %d loss: %.3f' % (epoch+1, running_loss/len(train_data)))

        training_loss = (running_loss / len(train_data))
        trainingLossPlot.append(float(round(training_loss, 4)))
        

        print("----------------")
        print("\n")
        
        print("\n")
        print("----------------")
        epoch+=1
        epochs_total = epochs_total - 1

    print('Model Trained')

    rmse_mean=statistics.mean(rmse)
    rmse_std=statistics.stdev(rmse)
   
    return trainingLossPlot, rmse_mean, rmse_std

def loss_variation(loss,epochs):
    for x in range(1,epochs):
        epochs.append(x)
    plt.xlabel('Number of Epoch')
    plt.ylabel('Training Loss')
    plt.title('Loss Variation')

    plt.plot(epochs, loss)
    plt.show()

class dataTrain(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X).to(torch.float)
        self.Y = torch.tensor(Y).to(torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.Y[idx]-1


data_train = dataTrain(train_data, label_train_data)
train_dataloader = DataLoader(data_train, batch_size=100, shuffle=True) #Shuffle option makes the data be trained patternless

# Train the model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=learningrate, momentum=0.9)
output=train()

print("\n")
print("RMSE Mean: ", (round(output[1], 4)))
print("\n")
print("RMSE SD: ", (round(output[2], 4)))
print("\n")

torch.save(model.state_dict(), 'my_model.pt')


#Plotting the learning curve
loss_variation(output[0], 100)


# Save the trained model

