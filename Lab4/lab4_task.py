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

def assign_clusters(X, centroids):
    """
    Assigns each data point to the nearest centroid.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        centroids (ndarray): Current centroids of shape (k, n_features).

    Returns:
        labels (ndarray): Cluster assignments for each data point.
    """

    X_3D = X[: , np.newaxis , :]
    dist = np.sum((X_3D - centroids) ** 2 , axis=2)
    labels = np.argmin(dist , axis=1)
    
    return labels


def update_centroids(X, labels, k):
    """
    Updates cluster centroids based on the mean of assigned points.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        labels (ndarray): Cluster assignments for each data point.
        k (int): Number of clusters.

    Returns:
        new_centroids (ndarray): Updated centroids of shape (k, n_features).
    """
    # TODO: Compute new centroids as the mean of assigned data points
    new_centroids = np.zeros((k , X.shape[1]))
    for i in range(k):
        mask = (labels == i)
        selected = X[mask]
        if selected.shape[0] == 0:
            new_centroid = new_centroids[i]
        else:
            new_centroid = np.sum(selected , axis=0) / selected.shape[0]
        new_centroids[i] = new_centroid


    return new_centroids

def k_means(X, k, max_iters=100, tol=1e-4):
    """
    Runs the K-means clustering algorithm.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iters (int): Maximum iterations.
        tol (float): Convergence tolerance.

    Returns:
        final_centroids (ndarray): Final cluster centroids.
        final_labels (ndarray): Final cluster assignments.
    """
    # Step 1: Initialize centroids using K-means++
    centroids = initialize_centroids_kmeans_pp(X, k)

    for _ in range(max_iters):
        # Step 2: Assign points to clusters
        labels = assign_clusters(X, centroids)

        # Step 3: Compute new centroids
        new_centroids = update_centroids(X, labels, k)

        # Step 4: Check for convergence (centroids do not change significantly)
        if np.linalg.norm(centroids - new_centroids) < tol:
            break
        
        centroids = new_centroids

    return centroids, labels

# ------------
# DEBUG

if __name__ == "__main__":
    k = 3  
    final_centroids, final_labels = k_means(X, k)


    plt.figure(figsize=(12, 6))
    
    plt.subplot(1 , 2 , 1)
    plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Length")
    plt.title("Visualization of K-Means Clustering (Sepal Features)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 2], X[:, 3], c=final_labels, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Length")
    plt.title("Visualization of K-Means Clustering (Petal Features)")
    # plt.legend()
    plt.grid(True)

    plt.show()


