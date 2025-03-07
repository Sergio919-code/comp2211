import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets



iris = datasets.load_iris()
X = iris.data
feature_names = iris.feature_names
df = pd.DataFrame(X , columns=feature_names)

def preprocess_iris(df):
    """ Preprocesses only petal features (more relevant for clustering). """
    X = np.array(df)
    X_mins = X.min(axis=0)
    X_maxs = X.max(axis=0)
    X_scaled = (X - X_mins) / (X_maxs - X_mins)
    
    return X_scaled

X = preprocess_iris(df)

# ------------------
# My Tasks

def initialize_centroids_kmeans_pp(X, k):
    """
    Initializes k cluster centroids using a simplified max-min method.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        k (int): Number of clusters.

    Returns:
        centroids (ndarray): Initialized centroids of shape (k, n_features).
    """    

    centroids = X[np.random.randint(0 , X.shape[0])]      #2D array
    X_3D = X[: , np.newaxis , :]   #change it to a 3D array for calculating distance with different centroid

    for _ in range(1 , k):
        Euclidean_dist = np.sum((X_3D - centroids) ** 2 , axis=2) #dist with each centriod , a 2D array
        nearest_dist = np.min(Euclidean_dist , axis=1)  #dist to it's cloests point
        probability = nearest_dist / np.sum(nearest_dist)
        
        next_index = np.random.choice(X.shape[0] , p=probability)
        next_centroid = X[[next_index]]
        centroids = np.vstack((centroids , next_centroid))


    return centroids

# ------------
# DEBUG
centroids = initialize_centroids_kmeans_pp(X , 5)
print(centroids)