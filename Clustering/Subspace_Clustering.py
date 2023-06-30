import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import kneighbors_graph
from torchvision.models import mobilenet_v2
from scipy.linalg import norm
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import matplotlib as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = mobilenet_v2(pretrained=True).features
model.eval()

# Set device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

DATA_PATH_TRAIN="train.csv"
DATA_PATH_TEST="test.csv"


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


def ssc(X, K=5, beta=1):
    """
    Subspace Clustering Scan algorithm
    
    Parameters:
    X: torch.tensor, (n_samples, n_features)
        Input data matrix
    K: int, default=5
        Number of nearest neighbors for the adjacency graph
    beta: float, default=1
        Regularization parameter
        
    Returns:
    labels: numpy.ndarray, (n_samples,)
        Cluster assignments for each data point
    """
    # Compute the affinity matrix
    W = kneighbors_graph(X, K, mode='connectivity', include_self=False).toarray()
    W = 0.5 * (W + W.T)  # make the affinity matrix symmetric
    
    # Compute the diagonal matrix D
    D = np.diag(np.sum(W, axis=1))
    
    # Compute the Laplacian matrix L
    L = D - W
    
    # Compute the subspace clustering matrix
    S = np.linalg.inv(X.T.dot(X) + beta * np.eye(X.shape[1])).dot(X.T).dot(L).dot(X).dot(np.linalg.inv(X.T.dot(X) + beta * np.eye(X.shape[1])))
    
    # Compute the spectral clustering matrix
    U, Sigma, Vt = np.linalg.svd(S, full_matrices=False)
    V = Vt.T
    V = V[:, :2]  # project to the first two principal components
    
    # Cluster the data using k-means
    centroids = np.random.randn(2, 2)
    for i in tqdm(range(10)):
        # Assign each data point to the closest centroid
        distances = np.linalg.norm(V[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update the centroids
        for j in tqdm(range(2)):
            centroids[j] = np.mean(V[labels == j], axis=0)
    
    return labels
features=get_features(train_data)

print(ssc(features))

cluster_labels = ssc(features)

# Compute the NMI between the cluster labels and true class labels
nmi = normalized_mutual_info_score(Y_train, cluster_labels)
print("NMI:", nmi)


# Compute t-SNE embedding of the features
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(features)

# Plot the t-SNE embedding with cluster labels
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels)
plt.title("t-SNE embedding with cluster labels")
plt.show()

# Compute t-SNE embedding of the features in 3D
tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(features)

# Plot the t-SNE embedding in 3D with cluster labels
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=cluster_labels)
ax.set_title("t-SNE embedding in 3D with cluster labels")
plt.show()