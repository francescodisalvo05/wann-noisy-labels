import torch.nn.functional as F
import numpy as np
import torch

from torch.utils.data import TensorDataset
from scipy import stats
from math import inf


def inject_noise(dataset, x, y, pattern, ratio):
    """
    Inject noise into the dataset based on the specified pattern and ratio.
    Args:
        dataset (str): The name of the dataset being used.
        x (np.ndarray): Feature vectors of the dataset.
        y (np.ndarray): Labels of the dataset.
        pattern (str): Type of noise to apply ('sym', 'asym', 'instance').
        ratio (float): Proportion of labels to corrupt with noise.
    Returns:
        np.ndarray: Noisy labels generated according to the specified pattern.
    """

    if pattern == "sym":
        T = get_sym_T(ratio, np.max(y)+1)
        y_noisy = create_noisy_labels(y, T)

    elif pattern == "asym":
        
        if dataset == "cifar10":
            T = get_asym_T_cifar10(ratio)
        elif dataset == "cifar100":
            T = get_asym_T_cifar100(ratio)
        elif dataset == "mnist":
            T = get_asym_T_mnist(ratio)
        elif dataset == "dermamnist":
            T = get_asym_T_dermamnist(ratio)

        y_noisy = create_noisy_labels(y, T)

    elif pattern == "instance":
        y_noisy = get_instance_noisy_label(ratio, x, y, norm_std=0.1)

    return y_noisy


def create_noisy_labels(labels, trans_matrix):
    """
    Generate noisy labels using a transition matrix.
    Args:
        labels (np.ndarray): Original labels.
        trans_matrix (np.ndarray): Transition matrix defining label corruption.
    Returns:
        np.ndarray: Noisy labels created from the original labels.
    """
    
    if trans_matrix is None:
        raise ValueError('Noisy matrix is None')
    
    num_trans_matrix = trans_matrix.copy()
    labels = labels.copy()
    
    num_classes = len(trans_matrix)
    class_idx = [np.where(np.array(labels) == i)[0]
                 for i in range(num_classes)]
    num_samples_class = [len(class_idx[idx])
                         for idx in range(num_classes)]
    for real_label in range(num_classes):
        for trans_label in range(num_classes):
            num_trans_matrix[real_label][trans_label] = \
                trans_matrix[real_label][trans_label] * num_samples_class[real_label]
    num_trans_matrix = num_trans_matrix.astype(int)

    for real_label in range(num_classes):
        for trans_label in range(num_classes):

            if real_label == trans_label:
                continue

            num_trans = num_trans_matrix[real_label][trans_label]
            if num_trans == 0:
                continue

            trans_samples_idx = np.random.choice(class_idx[real_label],
                                                 num_trans,
                                                 replace=False)
            class_idx[real_label] = np.setdiff1d(class_idx[real_label],
                                                 trans_samples_idx)
            for idx in trans_samples_idx:
                labels[idx] = trans_label
    
    return labels


def get_sym_T(eta, num_classes):
    """
    Generate a symmetric transition matrix.
    Args:
        eta (float): Noise rate (0 <= eta <= 1).
        num_classes (int): Total number of classes.
    Returns:
        np.ndarray: Symmetric transition matrix.
    """
    assert (eta >= 0.) and (eta <= 1.)
    
    diag_mask = np.eye(num_classes)
    rest_mask = 1 - diag_mask
    
    T = diag_mask * (1 - eta) \
        + rest_mask * eta / (num_classes - 1)
    
    return T


def get_asym_T_cifar10(eta):
    """
    Generate asymmetric transition matrix for CIFAR-10 dataset.
    Args:
        eta (float): Noise rate (0 <= eta <= 1).
    Returns:
        np.ndarray: Asymmetric transition matrix for CIFAR-10.
    """
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # truck -> automobile (9 -> 1)
    T[9, 9], T[9, 1] = 1. - eta, eta
    # bird -> airplane (2 -> 0)
    T[2, 2], T[2, 0] = 1. - eta, eta
    # cat <-> dog (3 <-> 5)
    T[3, 3], T[3, 5] = 1. - eta, eta
    T[5, 5], T[5, 3] = 1. - eta, eta
    # deer -> horse (4 -> 7)
    T[4, 4], T[4, 7] = 1. - eta, eta
    return T

def get_asym_T_cifar100(eta):
    """
    Generate asymmetric transition matrix for CIFAR-100 dataset.
    Args:
        eta (float): Noise rate (0 <= eta <= 1).
    Returns:
        np.ndarray: Asymmetric transition matrix for CIFAR-100.
    """
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 100
    num_superclasses = 20
    num_subclasses = 5

    T = np.eye(num_classes)

    for i in np.arange(num_superclasses):
        # build T for one superclass
        T_superclass = (1. - eta) * np.eye(num_subclasses)
        for j in np.arange(num_subclasses - 1):
            T_superclass[j, j + 1] = eta
        T_superclass[num_subclasses - 1, 0] = eta
        
        init, end = i * num_subclasses, (i + 1) * num_subclasses
        T[init:end, init:end] = T_superclass

    return T

def get_asym_T_mnist(eta):
    """
    Generate asymmetric transition matrix for MNIST dataset.
    Args:
        eta (float): Noise rate (0 <= eta <= 1).
    Returns:
        np.ndarray: Asymmetric transition matrix for MNIST.
    """
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # 7 -> 1
    T[7, 7], T[7, 1] = 1. - eta, eta
    # 2 -> 7
    T[2, 2], T[2, 7] = 1. - eta, eta
    # 5 <-> 6
    T[5, 5], T[5, 6] = 1. - eta, eta
    T[6, 6], T[6, 5] = 1. - eta, eta
    # 3 -> 8
    T[3, 3], T[3, 8] = 1. - eta, eta
    return T

def get_asym_T_dermamnist(eta):
    """
    Generate asymmetric transition matrix for DermaMNIST dataset.
    Args:
        eta (float): Noise rate (0 <= eta <= 1).
    Returns:
        np.ndarray: Asymmetric transition matrix for DermaMNIST.
    """
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)

    # 0 -> 1
    T[0, 0], T[0, 1] = 1. - eta, eta
    # 1 -> 2
    T[1, 1], T[1, 2] = 1. - eta, eta
    # 2 -> 3
    T[2, 2], T[2, 3] = 1. - eta, eta
    # 3 -> 4
    T[3, 3], T[3, 4] = 1. - eta, eta
    # 4 -> 5
    T[4, 4], T[4, 5] = 1. - eta, eta
    # 5 -> 6
    T[5, 5], T[5, 6] = 1. - eta, eta
    # 6 -> 0
    T[6, 6], T[6, 0] = 1. - eta, eta
    return T


def get_instance_noisy_label(n, x, y, norm_std): 
    """
    Generate instance-dependent noisy labels using feature-based corruption.
    source: https://github.com/xiaoboxia/Part-dependent-label-noise (NIPS, 2020)
    
    Args:
        n (float): Noise ratio.
        x (np.ndarray): Feature vectors.
        y (np.ndarray): Original labels.
        norm_std (float): Standard deviation of noise distribution (default 0.1).
    Returns:
        np.ndarray: Noisy labels generated by instance-dependent noise.
    """
    dataset = TensorDataset(torch.tensor(x).double(), torch.tensor(y))
    
    labels = y.copy()
    num_classes = np.max(labels) + 1
    feature_size = len(x[0])

    label_num = num_classes
    dataset = TensorDataset(torch.tensor(x).double(), torch.tensor(y))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    # labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.tensor(W).double()#.cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        # x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    
    # P = torch.stack(P, 0).cpu().numpy()
    P = torch.stack(P, 0).numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1

    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break

    return np.array(new_label)