


#USED FASHION-MNIST DATASET


import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


#Setting up parameter to use gpu 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


#Reading features and labels files as data frames
DATA_PATH_TRAIN="traindata.csv"
DATA_PATH_TEST="testdata.csv"

Xtrain = pd.read_csv(DATA_PATH_TRAIN,header=None, sep=",",usecols=list(range(1, 784)), skiprows=[0])
print(Xtrain)

Ytrain = pd.read_csv(DATA_PATH_TRAIN,header=None, sep=",", usecols=[0], skiprows=[0])
Ytrain = Ytrain.iloc[:, 0]
print(Ytrain)

Xtest = pd.read_csv(DATA_PATH_TEST, header=None, sep=",",usecols=list(range(1, 784)), skiprows=[0]) 
print(Xtest)

Ytest = pd.read_csv(DATA_PATH_TEST, header=None, sep=",", usecols=[0], skiprows=[0])  
Ytest = Ytest.iloc[:, 0]
print(Ytest)

#Converting our data frames to arrays
Xtrain=Xtrain.to_numpy()
Ytrain=Ytrain.to_numpy()

Xtest=Xtest.to_numpy()
Ytest=Ytest.to_numpy()

#Setting training parameters
X_test = Xtest.astype(np.int64)  
Y_test = Ytest.astype(np.int64) 

X_train = Xtrain.astype(np.int64)  
Y_train = Ytrain.astype(np.int64) 

class dataTrain(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(device), self.Y[idx].to(device)
def KNN(x_test, X_train, y_train, k):
    distances=[]
    for i, x_train in enumerate(X_train):
        dist = np.sqrt(np.sum((x_test - x_train) ** 2))
        distances.append((dist, y_train[i]))
    distances = sorted(distances, key=lambda x: x[0])
    k_neighbors = distances[:k]
    k_labels = []
    for i in range(len(k_neighbors)):
        k_labels.append(k_neighbors[i][1])

    most_common = max(set(k_labels), key=k_labels.count)
    return (most_common,)
def predictKNN(x_test, Y_train):
    predictions = []
    predictions.extend(KNN(x_test, X_train, Y_train, 3))
    predictions = np.array(predictions)
    return predictions

#Data Loader initialization
data_train = dataTrain(X_train, Y_train)

train_dataloader = DataLoader(data_train, batch_size=196, shuffle=True) #Shuffle option makes the data be trained patternless

mean = np.mean(X_train, axis=0)

X_train_centered = X_train - mean

covariance_matrix = np.cov(X_train_centered.T)

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

variance = eigenvalues / np.sum(eigenvalues)

cumu_variance = np.cumsum(variance)

X_train_pca = X_train_centered.dot(eigenvectors[:, :2])

print(X_train_pca)

# Choose a random data point from the test set
randvalue=random.randint(0, 783)
x_test_avalue = X_test[randvalue]
y_test_avalue = Y_test[randvalue]
print("Sample:", randvalue)

# Project the test point onto the first two principal components
x_test_pca = (x_test_avalue - mean).dot(eigenvectors[:, :2])

#KNN test on a point
distances=predictKNN(x_test_avalue,Y_train)
print("random label:", distances)

# Find the index of the nearest neighbor
index_of_nearest_neighbor = np.argmin(distances)

# Check if the nearest neighbor has the same label as the test point
y_pred_single = Y_train[index_of_nearest_neighbor]

print(y_pred_single)


y_pred=[]

for x in zip(X_test):
    y_pred.extend(predictKNN(x, Y_train))
    print(y_pred)
    
#Metric Calculations
y_test_binary = label_binarize(Y_test, classes=np.arange(10))

y_pred_prob = np.zeros((len(Y_test), 10))
for i in range(len(Y_test)):
    y_pred_prob[i, y_pred[i]] = 1
    print("y_pred_prob")

# Compute the accuracy, precision, recall, and F1 score
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred, average='weighted')
recall = recall_score(Y_test, y_pred, average='weighted')
f1 = f1_score(Y_test, y_pred, average='weighted')

# Compute the confusion matrix
confusion = confusion_matrix(Y_test, y_pred)

# Compute the ROC curve and AUC for each class and for all classes combined
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(1,10):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute the ROC curve and AUC
fpr_micro, tpr_micro, _ = roc_curve(y_test_binary.ravel(), y_pred_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("Confusion matrix:\n", confusion)

for i in range(1,10):
    print("Class", i, "AUC:", roc_auc[i])

print("Micro-average AUC:", roc_auc_micro)

# Compute the mean true positive rate and mean false positive rate
mean_tpr = np.mean(tpr, axis=0)
mean_fpr = np.linspace(0, 1, 100)

# Compute the AUC
macro_auc = auc(mean_fpr, mean_tpr)

# Plot ROC curve
plt.plot(mean_fpr, mean_tpr, color='b', label='Macro-average ROC curve (AUC = %0.2f)' % macro_auc)
plt.plot([0, 1], [0, 1], color='pink', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-average ROC Curve')
plt.legend(loc="lower right")
plt.show()