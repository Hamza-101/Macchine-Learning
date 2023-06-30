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
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

DATA_PATH_TRAIN="train.csv"


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


from scipy.spatial.distance import cdist
from scipy.linalg import eigvalsh


def spectral_clustering(X, k, sigma=None):
    
    n, p = X.shape
    
    # Compute affinity matrix
    if sigma is None:
        # Use median of pairwise Euclidean distances as sigma
        sigma = np.median(cdist(X, X, metric='euclidean'))
    A = np.exp(-cdist(X, X, metric='sqeuclidean') / (2 * sigma ** 2))
    
    # Compute degree matrix
    D = np.diag(np.sum(A, axis=1))
    
    # Compute Laplacian matrix
    L = D - A
    
    # Compute smallest k eigenvectors of L
    eigvals, eigvecs = eigvalsh(L, subset_by_index=[0, k-1])
    
    # Normalize eigenvectors by L2 norm
    Y = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
    
    # Cluster points using k-means on the rows of Y
    centroids, labels = k_means(Y, k)
    
    return labels
    
    
def k_means(X, k, max_iter=100):
    
    n, p = X.shape
    
    # Initialize centroids randomly
    idx = np.random.choice(n, k, replace=False)
    centroids = X[idx, :]
    
    for iter in tqdm(range(max_iter)):
        # Assign each point to the nearest centroid
        dist = cdist(X, centroids, metric='sqeuclidean')
        labels = np.argmin(dist, axis=1)
        
        # Update centroids as the mean of the assigned points
        for i in tqdm(range(k)):
            centroids[i, :] = np.mean(X[labels == i, :], axis=0)
            
    return centroids, labels
    

# Normalize data to have zero mean and unit variance
X_train_norm = (train_data - torch.mean(train_data, dim=0)) / torch.std(train_data, dim=0)

# Get deep features using pre-trained MobileNetV2 model
model = mobilenet_v2(pretrained=True).features
model.eval()

deep_features = get_features(X_train_norm)

# Run subspace clustering on deep features
k = 10  # Number of clusters
sigma = None  # Sigma for Gaussian kernel (use None for automatic selection)
labels = spectral_clustering(deep_features, k, sigma)

# Evaluate clustering using silhouette score
score = silhouette_score(deep_features, labels, metric='cosine')
print("Silhouette score:", score)


# Calculate NMI
nmi = normalized_mutual_info_score(Y_train, labels)
print("Normalized Mutual Information:", nmi)



# Run t-SNE on deep features
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(deep_features)

# Visualize data in 2D
plt.figure(figsize=(8, 8))
for i in range(k):
    plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1], label=f'Cluster {i}')
plt.legend()
plt.title('t-SNE Visualization in 2D')
plt.show()

# Run t-SNE on deep features again for 3D visualization
tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
X_tsne_3d = tsne.fit_transform(deep_features)

# Visualize data in 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(k):
    ax.scatter(X_tsne_3d[labels == i, 0], X_tsne_3d[labels == i, 1], X_tsne_3d[labels == i, 2], label=f'Cluster {i}')
ax.legend()
ax.set_title('t-SNE Visualization in 3D')
plt.show()