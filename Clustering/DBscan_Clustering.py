import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as plt
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from torchvision.models import mobilenet_v2
from sklearn.metrics.cluster import normalized_mutual_info_score

import torchvision
# # Set device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
def get_features(dataset):
    
    features = []
    with torch.no_grad():
        for images in tqdm(dataset):
            images = images.reshape(1, 1, 28, 28)
            images = images.long() 
           
            rgb_image = np.repeat(images, 3, axis=1)
            rgb_image = rgb_image.float() 
            
            deep_features = model(rgb_image)
            
            features.append(deep_features.numpy().flatten())
    
    return np.concatenate(features)


DATA_PATH_TRAIN="train.csv"
DATA_PATH_TEST="test.csv"

# Load the Fashion MNIST dataset
X_train = pd.read_csv(DATA_PATH_TRAIN,header=None, sep=",",usecols=list(range(1, 785)), skiprows=[0])
X_train=X_train.to_numpy()
X_train = X_train.astype(np.int64)  
train_data=torch.tensor(X_train)

Y_train = pd.read_csv(DATA_PATH_TRAIN,header=None, sep=",", usecols=[0], skiprows=[0])
Y_train = Y_train.iloc[:, 0]
Y_train=Y_train.to_numpy()
Y_train = Y_train.astype(np.int64) 
train_label=torch.tensor(Y_train)

model = mobilenet_v2(pretrained=True).features
model.eval()

def dist(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

def dbscan(X, eps, min_samples):
    """
    X: Input data
    eps: Epsilon, radius of the neighborhood to search
    min_samples: Minimum number of points in the epsilon-neighborhood to form a cluster
    """
    n = X.shape[0]
    visited = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=int)
    cluster_num = 0
    for i in tqdm(range(n)):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = []
        for j in range(n):
            if dist(X[i], X[j]) <= eps:
                neighbors.append(j)
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue
        cluster_num += 1
        labels[i] = cluster_num
        for j in neighbors:
            if visited[j]:
                continue
            visited[j] = True
            new_neighbors = []
            for k in range(n):
                if dist(X[j], X[k]) <= eps:
                    new_neighbors.append(k)
            if len(new_neighbors) >= min_samples:
                for k in new_neighbors:
                    if k not in neighbors:
                        neighbors.append(k)
            if labels[j] == 0:
                labels[j] = cluster_num
    return labels
def plot_clusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels.numpy(), cmap='tab10')
    plt.show()

if __name__ == '__main__':
    # Load Fashion MNIST dataset
    features=get_features(train_data)
    # Run DBSCAN
    eps = 10
    min_samples = 10
    cluster_labels = dbscan(features, eps, min_samples)
    
    # Plot clusters
    plot_clusters(features, cluster_labels)

 # Compute NMI
    nmi = normalized_mutual_info_score(Y_train, cluster_labels)
    print("NMI:", nmi)

    # Visualize clusters in 2D
    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(features)
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=cluster_labels, cmap='tab10')
    plt.title("DBSCAN Clustering with TSNE in 2D")
    plt.show()

    # Visualize clusters in 3D
    tsne = TSNE(n_components=3)
    features_tsne = tsne.fit_transform(features)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2], c=cluster_labels, cmap='tab10')
    ax.set_title("DBSCAN Clustering with TSNE in 3D")
    plt.show()