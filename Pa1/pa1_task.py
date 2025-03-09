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


if __name__ == "__main__":

    example_array = np.array(["CancerA", "Normal", "CancerB"])
    example_array = example_array.astype(object)
    result = label_to_integer(label=example_array)
    print(result)
    # You are expected to get [1,0,2]

    exit()

    train_feature = pd.read_csv("train_features.csv", index_col=0)
    train_label = pd.read_csv("train_labels.csv", index_col=0)
    test_feature = pd.read_csv("test_features.csv", index_col=0)
    train_feature = train_feature.to_numpy()
    train_label = train_label.to_numpy()
    test_feature = test_feature.to_numpy()
    train_label = train_label.flatten()

    print(train_feature.shape)
    print(train_label.shape)


