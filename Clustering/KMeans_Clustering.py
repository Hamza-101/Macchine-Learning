
import numpy as np
import torch
import random
from sklearn.metrics.pairwise import euclidean_distances
from torchvision.models import mobilenet_v2
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from PIL import Image
from tqdm import tqdm
from tensorflow import keras
import math
from tqdm import tqdm
import pandas as pd
import matplotlib as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import normalized_mutual_info_score



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_PATH_TRAIN="train.csv"
DATA_PATH_TEST="test.csv"

model = mobilenet_v2(pretrained=True).features
model.eval()

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

X_train = pd.read_csv(DATA_PATH_TRAIN,header=None, sep=",",usecols=list(range(1, 785)), skiprows=[0])
X_train=X_train.to_numpy()
X_train = X_train.astype(np.int64)  
train_data=torch.tensor(X_train)

Y_train = pd.read_csv(DATA_PATH_TRAIN,header=None, sep=",", usecols=[0], skiprows=[0])
Y_train = Y_train.iloc[:, 0]
Y_train=Y_train.to_numpy()
Y_train = Y_train.astype(np.int64) 
train_label=torch.tensor(Y_train)


deep_features = get_features(train_data)
print(deep_features)


#KNN


k = 10
centroids = np.array(random.sample(list(X_train), k))

# Create a dictionary to store the cluster assignments
cluster_assignments = {}

# Initialize the dictionary with empty lists for each cluster
for i in tqdm(range(k)):
    cluster_assignments[i] = []

# Define a function to compute Euclidean distance between two points
def euclidean_distance(x1, x2):
    return math.sqrt(sum((x1 - x2)**2))

# Define a function to assign data points to clusters
def assign_clusters(X, centroids):
    for i in range(len(X)):
        distances = [euclidean_distance(X[i], centroid) for centroid in centroids]
        cluster = distances.index(min(distances))
        cluster_assignments[cluster].append(X[i])

# Define a function to update centroids based on the mean of data points in each cluster
def update_centroids(X, centroids):
    for i in range(k):
        centroids[i] = np.mean(cluster_assignments[i], axis=0)

# Repeat the assign and update steps until convergence or a maximum number of iterations
max_iterations = 100
for i in tqdm(range(max_iterations)):
    assign_clusters(X_train, centroids)
    update_centroids(X_train, centroids)

# Print the cluster assignments for each data point
for i in tqdm(range(k)):
    print("Cluster {}: {} data points".format(i, len(cluster_assignments[i])))

# Compute the within-cluster sum of squares (WCSS) to evaluate performance
wcss = 0
for i in tqdm(range(k)):
    for j in range(len(cluster_assignments[i])):
        wcss += euclidean_distance(cluster_assignments[i][j], centroids[i])**2
print("WCSS:", wcss)


# Compute normalized mutual information (NMI) between the cluster labels and class labels
nmi = normalized_mutual_info_score(train_label, kmeans.labels_)
print("NMI:", nmi)

# Visualize data using t-SNE in 2D
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=0)
tsne_features = tsne.fit_transform(deep_features)

fig, ax = plt.subplots()
for i in range(k):
    ax.scatter(tsne_features[kmeans.labels_==i, 0], tsne_features[kmeans.labels_==i, 1], label="Cluster {}".format(i))
ax.legend()
plt.show()

# Visualize data using t-SNE in 3D
tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, n_iter=1000, random_state=0)
tsne_features = tsne.fit_transform(deep_features)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(k):
    ax.scatter(tsne_features[kmeans.labels_==i, 0], tsne_features[kmeans.labels_==i, 1], tsne_features[kmeans.labels_==i, 2], label="Cluster {}".format(i))
ax.legend()
plt.show()