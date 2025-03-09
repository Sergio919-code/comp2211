import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def calculate_rank(abundance_matrix):
    """
    Calculate the rank of protein abundance for each cell.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 2D numpy array of the same shape, containing the rank of protein abundance
    for each cell, where the lowest abundance is ranked 0.
    """
    sorted_ = np.argsort(abundance_matrix , axis=1)
    return sorted_

def calculate_mean(abundance_matrix):
    """
    Calculate the mean abundance of each protein across all cells.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 1D numpy array of shape (num_proteins,) containing the mean abundance.
    """
    sorted_ = np.sort(abundance_matrix , axis=1)
    mean = np.mean(sorted_ , axis=0)
    return mean

def substitute_mean(abundance_matrix, mean_values, rank_matrix):
    """
    Substitute each value in the abundance matrix with the corresponding mean value based on ranks.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.
    mean_values: A 1D numpy array of shape (num_proteins,) containing the mean abundance.
    rank_matrix: A 2D numpy array of shape (num_cells, num_proteins) representing the ranks
                 of protein abundances in each cell.

    Returns:
    A 2D numpy array of shape (num_cells, num_proteins) where each value has been
    substituted by the corresponding mean value based on the rank.
    """
    substituded = np.empty_like(abundance_matrix)
    substituded[np.arange(abundance_matrix.shape[0])[: , np.newaxis] , rank_matrix] = mean_values
    return substituded

def quantile_normalization(abundance_matrix):
    """
    Perform quantile normalization on a protein abundance matrix.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 2D numpy array of the same shape where each value has been substituted
    by the mean value of its corresponding rank across all cells.
    """
    ranking = calculate_rank(abundance_matrix)
    mean = calculate_mean(abundance_matrix)
    normalized = substitute_mean(abundance_matrix , mean , ranking)
    return normalized

def z_score_normalization(abundance_matrix):
    """
    Perform Z-score normalization on a protein abundance matrix.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 2D numpy array of the same shape where each value has been normalized
    using the Z-score formula: (X - mean) / std.
    """
    mean = np.mean(abundance_matrix , axis=0)
    std = np.std(abundance_matrix , axis=0 )
    normalized = (abundance_matrix - mean) / std
    return normalized

def preprocess_datasets(abundance_matrix):
    """
    Preprocess the protein abundance matrix by applying quantile normalization followed by Z-score normalization.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 2D numpy array of the same shape, representing the processed dataset after
    quantile normalization and Z-score normalization.
    """
    quantile_normalized = quantile_normalization(abundance_matrix)
    Z_normalized = z_score_normalization(quantile_normalized)
    return Z_normalized

def label_to_integer(label):
    """
    Convert string labels to integer labels.

    You may consider using np.where
    (https://numpy.org/doc/stable/reference/generated/numpy.where.html)

    Parameters:
    labels: A 1D numpy array of shape (num_cells,) containing string labels for each cell.

    Returns:
    A 1D numpy array of the same shape, where each string label has been converted
    to an integer: "Normal" -> 0, "CancerA" -> 1, "CancerB" -> 2.
    """
    mapped_label = np.where(label == "Normal" , 0 , np.where(label == "CancerA" , 1 , 2))
    return mapped_label

def PCA_and_visualization(abundance_matrix, label):
    """
    Perform PCA on the protein abundance matrix and visualize the results in a 2D scatter plot.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.
    labels: A 1D numpy array of shape (num_cells,) containing integer labels for each cell.

    Returns:
    x: The x-coordinates of the points in the scatter plot.
    y: The y-coordinates of the points in the scatter plot.
    colors: A list of colors corresponding to each label for visualization.
    component_number: The number of components you have kept.
    """
    """    
    Specify the number of components you would like to keep.
    Hint: We would need to visualize our dataset in a 2D scatter plot.
    Hint: You may try different numbers of component number and print out the result.
    """

    """
    You need to understand the meaning of input parameters x, y and c into function plt.scatter().
    Hint: You may check document as well as usage at https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    and https://matplotlib.org/stable/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py.
    Hint: You do not need to care about inputs other than x,y and c.
    Hint: You have obtained an array principal_component of shape (number of cells, component_number), and you would like to visualize it via a 2D scatter plot.
    The 2 values in each row of array principal_component represent the coordinate of a point.
    You may consider using np.where (https://numpy.org/doc/stable/reference/generated/numpy.where.html)
    """
    component_number = 2 

    pca = PCA(n_components=component_number, svd_solver="arpack", random_state=2)
    principal_component = pca.fit_transform(abundance_matrix)

    X = principal_component[: , 0]
    y = principal_component[: , 1]

    color_list = ["r", "b", "g", "c"]

    colors = np.array(color_list)[label.astype(int)]

    return X , y , colors , component_number

def visualize_processed_datasets(X, label):
    x, y, colors, _ = PCA_and_visualization(X, label)
    plt.scatter(x=x, y=y, c=colors)
    plt.show()


def calculate_manhattan_distance(feature_train, feature_test):
    """
    Calculate the Manhattan distance between training and testing feature matrices.

    You may consider using np.expand_dims https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    You may consider using np.sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    You may consider using np.abs https://numpy.org/doc/stable/reference/generated/numpy.abs.html

    Parameters:
    feature_train: A 2D numpy array of shape (num_cells_train, num_proteins)
                   representing protein abundance in the training set.
    feature_test: A 2D numpy array of shape (num_cells_test, num_proteins)
                  representing protein abundance in the testing set.

    Returns:
    A 2D numpy array of shape (num_cells_test, num_cells_train) representing the
    Manhattan distance between each test cell and each train cell.
    """

    X_TRAIN_3D = feature_train[np.newaxis , : ,:]
    X_TEST_3D = feature_test[: , np.newaxis , :]
    dist = np.sum(np.abs(X_TEST_3D - X_TRAIN_3D) , axis=2)

    return dist

def calculate_euclidean_distance(feature_train, feature_test):
    """
    Calculate the Euclidean distance between training and testing feature matrices.

    You may consider using np.expand_dims https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    You may consider using np.sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    You may consider using np.sqrt https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
    You may consider using np.square https://numpy.org/doc/stable/reference/generated/numpy.square.html

    Parameters:
    feature_train: A 2D numpy array of shape (num_cells_train, num_proteins)
                   representing protein abundance in the training set.
    feature_test: A 2D numpy array of shape (num_cells_test, num_proteins)
                  representing protein abundance in the testing set.

    Returns:
    A 2D numpy array of shape (num_cells_test, num_cells_train) representing the
    Euclidean distance between each test cell and each train cell.
    """
    X_TRAIN_3D = feature_train[np.newaxis , : ,:]
    X_TEST_3D = feature_test[: , np.newaxis , :]
    dist = np.sqrt(np.sum(np.square(X_TEST_3D - X_TRAIN_3D) , axis=2))

    return dist

def choose_nearest_neighbors(k, distance_metric, feature_train, feature_test, labels):
    """
    Choose the k nearest neighbors for each test cell based on the specified distance metric.

    You may consider using np.argsort https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
    You may consider using np.sort https://numpy.org/doc/stable/reference/generated/numpy.sort.html
    You may consider using np.take https://numpy.org/doc/stable/reference/generated/numpy.take.html

    Parameters:
    k: The number of nearest neighbors (integer).
    distance_metric: A string that can be either 'manhattan' or 'euclidean' indicating which distance metric to be used.
    feature_train: A 2D numpy array of shape (num_cells_train, num_proteins) representing protein abundance in the training set.
    feature_test: A 2D numpy array of shape (num_cells_test, num_proteins) representing protein abundance in the testing set.
    labels: A 1D numpy array of shape (num_cells_train,) containing labels of each cell in the training set.

    Returns:
    distance_k: A 2D numpy array of shape (num_cells_test, k) containing distances to the k nearest neighbors.
    top_k_labels: A 2D numpy array of shape (num_cells_test, k) containing labels of the k nearest neighbors.
    """

    if distance_metric == "manhattan":
        dist = calculate_manhattan_distance(feature_train , feature_test)   #(n_cell_test, n_cell_train)
    elif distance_metric == "euclidean":
        dist = calculate_euclidean_distance(feature_train , feature_test)
    else:
        raise TypeError("Unsupported metric type.")
    
    sorted_dist = np.sort(dist , axis=1)
    distance_k = sorted_dist[: , :k]

    sorted_index = np.argsort(dist , axis=1)
    index_k = sorted_index[: , :k]
    top_k_labels = labels[index_k]

    return distance_k , top_k_labels

def count_neighbor_class(top_k_labels):
    """
    Count the number of neighbors of each class among the k nearest neighbors for each test cell.

    You may consider using np.expand_dims https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    You may consider using np.sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    You may consider using np.arange https://numpy.org/doc/stable/reference/generated/numpy.arange.html

    Parameters:
    top_k_labels: A 2D numpy array of shape (num_cells_test, k) containing labels of the k nearest neighbors.

    Returns:
    class_count: A 2D numpy array of shape (num_cells_test, num_classes) representing the count of each class
                 among the k nearest neighbors for each test cell.
    """
    max_ = np.max(top_k_labels)
    identity_matrix = np.eye(max_ + 1)
    count = identity_matrix[top_k_labels]
    class_count = np.sum(count , axis=1)

    return class_count

def predict_labels(class_count):
    """
    Predict the label for each test cell based on the class counts of the k nearest neighbors.

    You may consider using np.argmax https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

    Parameters:
    class_count: A 2D numpy array of shape (num_cells_test, num_classes) representing the number of
                 data points belonging to each class among the k nearest neighbors.

    Returns:
    predicted_labels: A 1D numpy array of shape (num_cells_test,) containing the predicted label
                      for each test cell.
    """
    pred_label = np.argmax(class_count , axis=1)

    return pred_label

if __name__ == "__main__":

    example_count = np.array([[2, 1, 0, 1], [2, 1, 1, 0]])
    result = predict_labels(class_count=example_count)
    print(result)
    # You are expected to get [0,0]
    exit()


    train_feature = pd.read_csv("train_features.csv", index_col=0)
    train_label = pd.read_csv("train_labels.csv", index_col=0)
    test_feature = pd.read_csv("test_features.csv", index_col=0)
    train_feature = train_feature.to_numpy()
    train_label = train_label.to_numpy()
    test_feature = test_feature.to_numpy()
    train_label = train_label.flatten()

    processed_train_feature = preprocess_datasets(abundance_matrix=train_feature)
    processed_test_feature = preprocess_datasets(abundance_matrix=test_feature)
    train_label = label_to_integer(label=train_label)

    #visualize_processed_datasets(processed_train_feature, train_label)


