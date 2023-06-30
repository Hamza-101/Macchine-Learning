import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm
from torchvision.models import mobilenet_v2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

deep_features=get_features(train_data)
dep_features=torch.tensor(deep_features)
# Define bandwidth parameter
bandwidth = 5

# Convert deep features to PyTorch tensor
deep_features = torch.from_numpy(deep_features).to(device)
deep_features = deep_features.reshape(-1, 1)
# Define mean shift clustering function
def mean_shift_clustering(X, bandwidth):
    """
    Implementation of mean shift clustering algorithm
    """
    # Define stopping criterion
    max_iterations = 100
    tol = 1e-4

    # Initialize centroids as input data points
    centroids = X.clone()

    # Iterate until convergence or maximum iterations reached
    for i in range(max_iterations):
        # Compute pairwise Euclidean distances between data points and centroids
        distances = cdist(X.cpu().numpy(), centroids.cpu().numpy())

        # Compute Gaussian kernel weights
        weights = np.exp(-distances**2 / (2 * bandwidth**2))

        # Compute weighted mean shift vectors
        shifts = np.sum(weights[:, :, np.newaxis] * (X.cpu().numpy()[:, np.newaxis, :] - centroids.cpu().numpy()), axis=0) \
                 / np.sum(weights[:, :, np.newaxis], axis=0)

        # Update centroids
        old_centroids = centroids.clone()
        centroids += torch.from_numpy(shifts).to(device)

        # Check for convergence
        if torch.allclose(centroids, old_centroids, rtol=tol):
            break

    # Assign data points to clusters based on nearest centroid
    distances = cdist(X.cpu().numpy(), centroids.cpu().numpy())
    labels = np.argmin(distances, axis=1)

    return labels

# Perform mean shift clustering on deep features
labels = mean_shift_clustering(deep_features, bandwidth)

# Print number of clusters found
num_clusters = len(np.unique(labels))
print("Number of clusters:", num_clusters)

nmi = normalized_mutual_info_score(Y_train, labels)
print("NMI:", nmi)

# Plot clusters
# plot_clusters(deep_features, labels)

nmi = normalized_mutual_info_score(Y_train, labels)
print("NMI:", nmi)

# Visualize clusters in 2D
tsne = TSNE(n_components=2)
features_tsne = tsne.fit_transform(deep_features)
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10')
plt.title("DBSCAN Clustering with TSNE in 2D")
plt.show()

# Visualize clusters in 3D
tsne = TSNE(n_components=3)
features_tsne = tsne.fit_transform(deep_features)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2], c=labels, cmap='tab10')
ax.set_title("DBSCAN Clustering with TSNE in 3D")
plt.show()