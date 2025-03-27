from src.model.utils import *

from sklearn.neighbors import NearestNeighbors

import numpy as np



class WANN():
    """
    Weighted Adaptive Nearest Neighbor (WANN) implementation for classification tasks.
    """

    def __init__(self, kmin=11, kmax=51):
        """
        Initialize the WANN model.
        Args:
            kmin (int): Minimum value of k for searching the optimal neighborhood size.
            kmax (int): Maximum value of k for searching the optimal neighborhood size.
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
        Predict labels for the given test data using weighted adaptive prediction.
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

        # Map classes in the neighborhoods
        neigh_classes = np.apply_along_axis(lambda arr: self.y[arr], axis=1, arr=indices)

        # Map from k to reliability_score
        mapper_weight = np.vectorize(lambda arr : 1.0/self.optimal_k[arr])
        curr_optimal_k = self.optimal_k[indices[:,0].copy()] 
        neigh_reliability_scores = mapper_weight(indices[:,:kmax])

        # Predict according to the weighted adaptive neighborhood
        predictions = np.array([adaptive_weighted_prediction(nc, nw, curr_optimal_k[idx]) for idx,(nc,nw) in enumerate(zip(neigh_classes,neigh_reliability_scores))])
        
        return predictions

    
    def find_k(self, x_train, y_train, kmin=11, kmax=51):
        """
        Find the optimal value of k for each training observation.
        Args:
            X (np.ndarray): Training data embeddings.
            y (np.ndarray): Training labels.
        Returns:
            np.ndarray: Array of optimal k values for each training observation.
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