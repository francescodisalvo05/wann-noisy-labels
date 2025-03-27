from collections import defaultdict
import numpy as np
import torch
import json
import os 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Loss setup

def get_config_file(dataset, name, root="assets/config"):
    """
    Load config file for the selected loss function.
    We mainly used open-source code and hyperparameters from:
        https://github.com/Virusdoll/Active-Negative-Loss/blob/main/loss.py (NIPS 2023)
    However, since it was not available the implementation/hyperparams of ELR, 
    we included the original implementation and relative suggested hyperparameters:
        https://github.com/shengliu66/ELR (NIPS 2020)

    Those hyperparameters were heavily tuned for each proposed datasets. Thus, for 
    our datasets that were not included on the proposed ones (e.g. medmnist), 
    we hardcoded the hyperparameters of "similar" available ones. Specifically:
        cifar-10n = cifar10
        cifar-100n = cifar100
        animal-10n = cifar10
        organcmnist = cifar10
        dermamnist = cifar10

    Args:
        dataset (str): Name of the dataset being used.
        name (str): Name of the loss function.
        root (str): Root directory where config files are stored (default is 'assets/config').

    Returns:
        dict: A dictionary containing the hyperparameters for the specified loss function.
    """
    
    # Handle cifar10n & cifar100n: use the hyperparams obtained for cifar10 and cifar100
    if 'cifar100' in dataset:
        dataset = 'cifar100'
    elif 'cifar10' in dataset:
        dataset = 'cifar10'
    
    # Handle medmnist & animal-10n: use the hyperparams obtained for cifar10
    elif dataset in ['breastmnist','dermamnist','animal-10n']:
        dataset = 'cifar10'

    config_path = os.path.join(root,f'{dataset}_{name}.json')

    with open(config_path, 'r') as f:
        config = json.load(f)
        return config['loss']


def get_loss(dataset, name, num_classes):
    """
    Helper function which initialize the respective robust loss function.
    Adapted from: https://github.com/Virusdoll/Active-Negative-Loss/blob/main/loss.py

    Args:
        dataset (str): Dataset name used for the current experiment.
        name (str): Name of the loss function to be used.
        num_classes (int): Number of classes for classification.
    Returns:
        dict: Configuration settings for the specified loss function.
    """
    config = get_config_file(dataset, name)

    # Loss
    if name == 'ce':
        from src.model.loss import ce
        loss = ce()
    elif name == 'fl':
        from src.model.loss import fl
        loss = fl(config)
    elif name == 'mae':
        from src.model.loss import mae
        loss = mae(num_classes)
    elif name == 'sce':
        from src.model.loss import sce
        loss = sce(num_classes, config)
    elif name == 'gce':
        from src.model.loss import gce
        loss = gce(num_classes, config)
    
    # Active Passive Loss
    elif name == 'nce_mae':
        from src.model.loss import nce_mae
        loss = nce_mae(num_classes, config)
    elif name == 'nce_rce':
        from src.model.loss import nce_rce
        loss = nce_rce(num_classes, config)
    elif name == 'nfl_rce':
        from src.model.loss import nfl_rce
        loss = nfl_rce(num_classes, config)
    
    # Asymmetric Loss
    elif name == 'nce_agce':
        from src.model.loss import nce_agce
        loss = nce_agce(num_classes, config)
    elif name == 'nce_aul':
        from src.model.loss import nce_aul
        loss = nce_aul(num_classes, config)
    elif name == 'nce_ael':
        from src.model.loss import nce_ael
        loss = nce_ael(num_classes, config)
    
    # Active Negative Loss
    elif name == 'anl_ce':
        from src.model.loss import anl_ce
        loss = anl_ce(num_classes, config)
    elif name == 'anl_fl':
        from src.model.loss import anl_fl
        loss = anl_fl(num_classes, config)
    elif name == 'anl_ce_er':
        from src.model.loss import anl_ce_er
        loss = anl_ce_er(num_classes, config)
    # Error
    else:
        raise ValueError('loss name error')
    
    return loss


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Utils function for both ANN and WANN. 
def mapper_train(arr, y):
    """
    Map indices to their corresponding labels.
    Args:
        arr (np.ndarray): Array of indices referring to training samples.
        y (np.ndarray): Array of training labels.
    Returns:
        np.ndarray: Array of labels corresponding to the given indices.
    """
    return y[arr]


def most_frequent(arr):
    """
    Determine the most frequently occurring element in the array.
    Args:
        arr (np.ndarray): Array containing class labels.
    Returns:
        int: The most frequently occurring class label.
    """
    unique_elements, counts = np.unique(arr, return_counts=True)
    most_frequent_element = unique_elements[np.argmax(counts)]
    return most_frequent_element


def adaptive_prediction(neighborhood, optimal_k):
    """
    Predict label based on the most frequent label within the optimal neighborhood.
    Args:
        neighborhood (np.ndarray): Array of labels in the neighborhood.
        optimal_k (int): Optimal number of neighbors to consider.
    Returns:
        int: Predicted label based on the neighborhood.
    """
    unique_elements, counts = np.unique(neighborhood[:optimal_k], return_counts=True)
    most_frequent_element = unique_elements[np.argmax(counts)]
    return most_frequent_element


def adaptive_weighted_prediction(neighborhood, weights, optimal_k):
    """
    Predict label using weighted voting within the optimal neighborhood.
    Args:
        neighborhood (np.ndarray): Array of labels in the neighborhood.
        weights (np.ndarray): Array of reliability scores corresponding to the labels.
        optimal_k (int): Optimal number of neighbors to consider.
    Returns:
        int: Predicted label based on weighted voting.
    """
    vote_counts = defaultdict(float)
    for cls_id, weight in zip(neighborhood[:optimal_k],weights[:optimal_k]):
        vote_counts[cls_id] += weight
    final_prediction = max(vote_counts, key=vote_counts.get)
    return final_prediction