import pytest
from lab4_task import *
from sklearn.datasets import make_blobs

def test_preprocess_iris():
    iris = datasets.load_iris()
    X = iris.data
    feature_names = iris.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    X_scaled = preprocess_iris(df)
    
    assert X_scaled.shape == X.shape
    assert np.all(X_scaled >= 0) and np.all(X_scaled <= 1)

def test_initialize_centroids_kmeans_pp():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=4, random_state=42)
    k = 3
    centroids = initialize_centroids_kmeans_pp(X, k)
    
    assert centroids.shape == (k, X.shape[1])

def test_assign_clusters():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=4, random_state=42)
    k = 3
    centroids = initialize_centroids_kmeans_pp(X, k)
    labels = assign_clusters(X, centroids)
    
    assert labels.shape == (X.shape[0],)
    assert np.all(labels >= 0) and np.all(labels < k)

def test_update_centroids():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=4, random_state=42)
    k = 3
    centroids = initialize_centroids_kmeans_pp(X, k)
    labels = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, labels, k)
    
    assert new_centroids.shape == (k, X.shape[1])

def test_k_means():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=4, random_state=42)
    k = 3
    final_centroids, final_labels = k_means(X, k)
    
    assert final_centroids.shape == (k, X.shape[1])
    assert final_labels.shape == (X.shape[0],)
    assert np.all(final_labels >= 0) and np.all(final_labels < k)

if __name__ == "__main__":
    pytest.main()