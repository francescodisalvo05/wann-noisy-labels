
from src.utils.noise import inject_noise
from src.model.WANN import WANN

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch


def preprocessing_knn(args, x_train, y_train, x_test, y_test):
    """
    Pre-processing pipeline for k-NN experiments. Includes options for dimensionality reduction, 
    noise injection, imbalance handling, and subsampling.
    Args:
        args (Namespace): Parsed command line arguments.
        x_train (np.ndarray): Training embeddings.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test embeddings.
        y_test (np.ndarray): Test labels.
    Returns:
        tuple: Processed training and testing datasets (x_train, y_train, x_test, y_test).
    """

    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Handle imbalance, if required
    if args.imbalance_ratio:
        x_train, y_train = produce_imbalance_data(args.imbalance_ratio, x_train.copy(),y_train.copy())

    # Subsample if required
    if args.num_subsamples > 0:
        x_train, y_train = resample_arrays_per_samples(x_train.copy(),y_train.copy(),args.num_subsamples, seed=args.seed) # extract the same number of samples for the val set

    # Inject noise if requested
    if args.noise_pattern != "None":
        y_train = inject_noise(args.dataset, x_train.copy(), y_train.copy(), args.noise_pattern, args.noise_ratio)

    # Normalize data
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Apply dimensionality reduction if requested
    if args.dimensionality_reduction != "None":
        x_train, x_test = dimensionality_reduction_pipeline(args, x_train, y_train, x_test)

    return x_train, y_train, x_test, y_test


def preprocess_probing(args, x_trainval, y_trainval, x_test=None, y_test=None):
    """
    Pre-processing pipeline for the linear probing experiments. Includes options for dimensionality reduction, 
    noise injection, imbalance handling, and subsampling.

    Args:
        args (Namespace): Parsed command line arguments.
        x_trainval (np.ndarray): Training embeddings.
        y_trainval (np.ndarray): Training labels.
        x_test (np.ndarray): Test embeddings.
        y_test (np.ndarray): Test labels.
    Returns:
        tuple: Data loaders for training, validation, and optionally testing.
    """

    num_classes = len(np.unique(y_trainval))
    
    # Subsample if required, and handle validation set
    if args.num_subsamples > 0:
        x_trainval, y_trainval = resample_arrays_per_samples(x_trainval, y_trainval, args.num_subsamples, seed=args.seed, additional_val=True)
        val_size = 25 * num_classes # extra 25 samples per class
    else:
        val_size = 0.15

    # Inject noise if needed
    if args.noise_pattern != "None":
        y_trainval = inject_noise(args.dataset, x_trainval.copy(), y_trainval.copy(), args.noise_pattern, args.noise_ratio)

    # split into train and val
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, 
                                                      test_size=val_size, 
                                                      stratify=y_trainval)

    # Normalize training and validation sets
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    # Convert to PyTorch tensors
    x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
    x_val, y_val = torch.Tensor(x_val), torch.Tensor(y_val)

    # Create datasets
    train_ds = CustomDataset(x_train, y_train)
    val_ds = CustomDataset(x_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=6, prefetch_factor=6, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=6, prefetch_factor=6, persistent_workers=True, pin_memory=True)

    # Handle test set if provided
    if x_test is not None:
        x_test = (x_test - mean) / std
        x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)
        test_ds = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=6, prefetch_factor=6, persistent_workers=True, pin_memory=True)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def preprocess_probing_medmnist(args, x_train, y_train, x_val, y_val, x_test=None, y_test=None):
    """
    Pre-processing pipeline specifically for MedMNIST datasets.
    Args:
        args (Namespace): Parsed command line arguments.
        x_train, y_train (np.ndarray): Training embeddings and labels.
        x_val, y_val (np.ndarray): Validation embeddings and labels.
        x_test, y_test (np.ndarray, optional): Test embeddings and labels.
    Returns:
        tuple: Data loaders for training, validation, and optionally testing.
    """

    # Inject noise if needed
    if args.noise_pattern != "None":
        y_train = inject_noise(args.dataset, x_train.copy(), y_train.copy(), args.noise_pattern, args.noise_ratio)
        y_val = inject_noise(args.dataset, x_val.copy(), y_val.copy(), args.noise_pattern, args.noise_ratio)
    
    # Normalize
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    
    # Convert to PyTorch tensors
    x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
    x_val, y_val = torch.Tensor(x_val), torch.Tensor(y_val)

    # Create datasets
    train_ds = CustomDataset(x_train, y_train)
    val_ds = CustomDataset(x_val, y_val)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Handle test set if provided
    if x_test is not None:
        x_test = (x_test - mean) / std
        x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)
        test_ds = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def resample_arrays_per_samples(x_trainval, y_trainval, num_subsamples, seed, additional_val = False):
    """
    Extract a stratified number of samples from training data.
    Args:
        x_trainval (np.ndarray): Combined training and validation embeddings.
        y_trainval (np.ndarray): Corresponding labels for embeddings.
        num_subsamples (int): Number of samples to extract.
        seed (int): Random seed for reproducibility.
        additional_val (bool): Whether to include additional validation samples.
    Returns:
        tuple: Resampled training embeddings and labels.
    """

    if additional_val:
        num_classes = len(np.unique(y_trainval))
        train_size = num_subsamples + 25 * num_classes
    else:
        train_size = num_subsamples

    x_sample, _, y_sample, _ = train_test_split(x_trainval, y_trainval, 
                                                train_size=train_size,  
                                                stratify=y_trainval,
                                                random_state=seed)
    
    return x_sample, y_sample


def produce_imbalance_data(imbalance_ratio, x_train, y_train):
    """
    Create an imbalanced dataset by reducing samples for certain classes based on an imbalance ratio.
    Code adapted from https://github.com/ynu-yangpeng/GLMC (CVPR 2023).
    Args:
        imbalance_ratio (float): Ratio determining the imbalance in the dataset.
        x_train (np.ndarray): Training embeddings.
        y_train (np.ndarray): Training labels.
    Returns:
        tuple: Imbalanced training data (x_train_lt, y_train_lt).
    """

    num_classes = len(set(y_train))
    data_percent = []
    data_num = int(x_train.shape[0] / num_classes)

    for cls_idx in range(num_classes):
        num = data_num * (imbalance_ratio ** (cls_idx / (num_classes - 1)))
        data_percent.append(int(num))

    print("Imbalance_ratio is {}".format(data_percent[0] / data_percent[-1]))
    print("Per class num：{}".format(data_percent))

    x_train_lt, y_train_lt = None, None

    for i in range(1, num_classes + 1):
        a1 = y_train >= i - 1
        a2 = y_train < i
        index = a1 & a2

        task_train_x = x_train[index]
        label = y_train[index]
        data_num = task_train_x.shape[0]
        index = np.random.choice(data_num, data_percent[i - 1],replace=False)
        tem_data = task_train_x[index]
        tem_label = label[index]

        if x_train_lt is None:
            x_train_lt = tem_data
            y_train_lt = tem_label
        else:
            x_train_lt = np.concatenate([x_train_lt, tem_data], axis=0)
            y_train_lt = np.concatenate([y_train_lt, tem_label], axis=0)

    return x_train_lt, y_train_lt


def dimensionality_reduction_pipeline(args, x_train, y_train, x_test):
    """
    Apply dimensionality reduction techniques such as PCA, LDA, or fLDA.
    Args:
        args (Namespace): Parsed command line arguments.
        x_train (np.ndarray): Training embeddings.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test embeddings.
    Returns:
        tuple: Reduced dimensionality training and testing embeddings.
    """

    if args.dimensionality_reduction == "PCA":
        pca = PCA(n_components=200, random_state=4524234) # fixed to 200 for each experiment | pca seed independent
        pca.fit(x_train)
        x_train_red = pca.transform(x_train)
        x_test_red = pca.transform(x_test)

    elif args.dimensionality_reduction == "LDA":
        lda = LinearDiscriminantAnalysis(solver='svd').fit(x_train,y_train)
        x_train_red = lda.transform(x_train)
        x_test_red = lda.transform(x_test)
    
    elif args.dimensionality_reduction == "fLDA":
        # need to run WANN to get the optimal k values
        wann = WANN(kmin=11, kmax=51)
        wann.fit(x_train,y_train)
        # filter out the samples having optimal k == 51 (i.e. kmax)
        x_train_to_fit = x_train[np.where(wann.optimal_k < 51)[0]].copy()
        y_train_to_fit = y_train[np.where(wann.optimal_k < 51)[0]].copy()
        # use the filtered samples to feed LDA
        lda = LinearDiscriminantAnalysis(solver='svd').fit(x_train_to_fit,y_train_to_fit)
        # transform the whole train and test set
        x_train_red = lda.transform(x_train)
        x_test_red = lda.transform(x_test)

    return x_train_red, x_test_red



class CustomDataset(Dataset):
    """
    A custom Dataset class to hold data, labels, and indices for PyTorch Dataloaders.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        # Return both the sample and its ID (idx in this case)
        return sample, label, idx