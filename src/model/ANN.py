from src.model.utils import adaptive_prediction, mapper_train, most_frequent

from sklearn.neighbors import NearestNeighbors

import numpy as np


class ANN():
    """
    Adaptive Nearest Neighbor (ANN) implementation for classification tasks.
    """
    def __init__(self, kmin=11, kmax=51):
        """
        Initialize the ANN model.
        Args:
            kmin (int): Minimum value of k to start the search.
            kmax (int): Maximum value of k to end the search.
        """
        self.kmin = kmin        
        self.kmax = kmax        
    
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        Args:
            X (np.ndarray): Training data embeddings.
            y (np.ndarray): Training labels.
        """
        self.unique_classes = list(set(y))
        self.num_classes = len(self.unique_classes)
        self.X = X
        self.y = y
        self.optimal_k = self.find_k(self.X, self.y, kmin=self.kmin, kmax=self.kmax)

        
    def predict(self, X):
        """
        Predict labels for the given test data.
        Args:
            X (np.ndarray): Test data embeddings.
        Returns:
            np.ndarray: Predicted labels.
        """
        x_test = np.array(X)
        kmax = np.max(self.optimal_k)

        # Get the closest training points for each test observation
        nn = NearestNeighbors(n_neighbors=kmax) 
        nn.fit(self.X)
        _, indices = nn.kneighbors(x_test)

        # Assign at each test observation the optimal value of k of the relative closest training observation
        # ... reliability score = 1 / k
        curr_optimal_k = self.optimal_k[indices[:,0].copy()] 

        # Map the ids of each neighborhood to the relative classes
        neigh_classes = np.apply_along_axis(lambda arr: self.y[arr], axis=1, arr=indices)

        # Predict labels based on the adaptive neighborhood 
        predictions = np.array([adaptive_prediction(nc, curr_optimal_k[idx]) for idx,nc in enumerate(neigh_classes)])

        return predictions

    
    def find_k(self, x_train, y_train, kmin=11, kmax=51):
        """
        Find the optimal value of k for each training sample.
        Args:
            X (np.ndarray): Training data embeddings.
            y (np.ndarray): Training labels.
        Returns:
            np.ndarray: Optimal k for each training observation.
        """

        # Initialize optimal_k with -1
        # Use -1 as flag to spot samples without a given optimal k, yet
        optimal_k = np.array([-1] * len(x_train))

        # First sample is the sample itself
        # Thus, check from 1 to kmax+1
        nn = NearestNeighbors(n_neighbors=kmax + 1) 
        nn.fit(x_train)
        _, indices = nn.kneighbors(x_train)
        
        # Increase by two in order to always get odd neighbhors
        for k in range(kmin,kmax,2):
            
            # Samples without an opimal k
            mask = np.array(np.where(optimal_k == -1)[0]) 

            # Exit condition: they all have an optimal k
            if len(indices[mask,1:k+1]) == 0:
                break
            
            # Map the ids with the classes in the neighborhood & predict the label
            k_classes = np.apply_along_axis(mapper_train, axis=1, arr=indices[mask,1:k+1], y=y_train)
            predictions = np.apply_along_axis(most_frequent, axis=1, arr=k_classes)
            
            # Update the ones that are correctly predicted
            correct_ids = np.array(np.where(predictions == y_train[mask])[0])
            optimal_k[mask[correct_ids]] = k
        
        # Edge case: hardcode k_max
        mask = np.array(np.where(optimal_k == -1)[0])
        optimal_k[mask] = kmax

        return optimal_k