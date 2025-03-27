
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNN():
    """
    A simple wrapper class for K-Nearest Neighbors (KNN) to standardize 
    the inference pipeline with ANN and WANN models.
    """
    def __init__(self, k = 11):
        """
        Initialize the KNN model.
        Args:
            k (int): Number of neighbors to use for prediction.
        """
        self.k = k
    
    
    def fit(self, X, y):
        """
        Fit the KNN model to the training data.
        Args:
            X (np.ndarray): Training data embeddings.
            y (np.ndarray): Training labels.
        """
        self.unique_classes = list(set(y))
        self.num_classes = len(self.unique_classes)
        self.X = np.array(X)
        self.y = np.array(y)

        
    def predict(self, X):
        """
        Predict labels for the given test data.
        Args:
            X (np.ndarray): Test data embeddings.
        Returns:
            np.ndarray: Predicted labels.
        """
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(self.X, self.y)
        y_pred = knn.predict(X)
        return y_pred