import torchvision.transforms as T
import torchvision
import numpy as np
import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.cifarn import CIFAR10N, CIFAR100N
from src.dataset.animal10 import ANIMAL10N

import torch
import random


def seed_worker(worker_id):
    """
    Ensure reproducibility by seeding worker processes.
    Args:
        worker_id (int): ID of the worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def extract_embeddings(model, device, dataloader):
    """
    Perform inference to extract embeddings using the provided model.
    Args:
        model (torch.nn.Module): Pre-trained model used for feature extraction.
        device (torch.device): Device to run the inference on (CPU or GPU).
        dataloader (DataLoader): DataLoader for the dataset (train/test).
    Returns:
        dict: Dictionary containing extracted embeddings and labels.
    """
    embeddings_db, labels_db = [], []

    for images, labels in tqdm(dataloader):
        images = images.to(device)

        output = model.forward_features(images)
        output = model.forward_head(output, pre_logits=True)

        labels_db.extend(labels)
        embeddings_db.extend(output.detach().cpu().numpy())

    return {'embeddings': embeddings_db, 'labels': labels_db}



def get_dataloaders(args, transforms, g):
    """
    Generate training and testing DataLoaders for the specified dataset.
    Args:
        args (Namespace): Parsed command line arguments.
        transforms (Transform): Data augmentation and transformation functions.
        g (Generator): PyTorch generator for reproducibility.
    Returns:
        dict: Dictionary containing DataLoaders for training and testing.
    """

    if args.dataset == "cifar10":
        trainds, testds = get_cifar10(args, transforms)
    elif args.dataset == "cifar100":
        trainds, testds = get_cifar100(args, transforms)
    elif args.dataset == "mnist":
        trainds, testds = get_mnist(args, transforms)
    elif args.dataset == "fashion-mnist":
        trainds, testds = get_fmnist(args, transforms)
    elif args.dataset.startswith('cifar10-n'):
        trainds, testds = get_cifar10n(args, transforms)
    elif args.dataset.startswith('cifar100-n'):
        trainds, testds = get_cifar100n(args, transforms)
    elif args.dataset == 'animal-10n':
        trainds, testds = get_animal10n(args, transforms)
    else:
        raise ValueError(f"{args.dataset} not available")
    
    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=False, num_workers=16,
                             persistent_workers=True, generator=g, worker_init_fn=seed_worker)
    testloader = DataLoader(testds, batch_size=args.batch_size, shuffle=False, num_workers=16,
                            persistent_workers=True, generator=g, worker_init_fn=seed_worker)

    return {'train': trainloader, 'test': testloader}


def get_databases(args):
    """
    Load pre-generated embeddings and labels from stored .npz files.
    Args:
        args (Namespace): Parsed command line arguments.
    Returns:
        tuple: Train and test embeddings and labels.
    """

    if args.dataset.startswith('cifar100-n') or args.dataset.startswith('cifar10-n'):
        dataset = args.dataset.split("-")[0] # load embeddings from cifar10 or cifar100
        noisy_labels = get_noisy_labels_cifarn(args)
    else:
        dataset = args.dataset

    db_path = os.path.join(args.database_root, 
                           dataset, 
                           args.backbone)    
    
    train_path = os.path.join(db_path,'train.npz')
    train_data = np.load(train_path)

    x_train = train_data['embeddings']
    y_train = train_data['labels'].astype(int)

    test_path = os.path.join(db_path,'test.npz')
    test_data = np.load(test_path)

    x_test = test_data['embeddings']
    y_test = test_data['labels'].astype(int)

    if args.dataset.startswith('cifar100-n') or args.dataset.startswith('cifar10-n'):
        return x_train, noisy_labels, x_test, y_test
    

    return x_train, y_train, x_test, y_test


def get_noisy_labels_cifarn(args):
    """
    Load noisy labels for CIFAR-N datasets (CIFAR-10N or CIFAR-100N).
    Args:
        args (Namespace): Parsed command line arguments containing dataset details.
    Returns:
        np.ndarray: Array of noisy labels corresponding to the training data.
    """
    if args.dataset.startswith('cifar10-n'):
        _, noise_type = args.dataset.split("/")
        root_path = os.path.join(args.dataset_root,"cifarn")
        trainset = CIFAR10N(root=root_path, download=True,  train=True, transform = None, noise_type = noise_type,
                            noise_path = os.path.join(root_path,'CIFAR-10_human.pt'), is_human = True)
        noisy_labels = trainset.train_noisy_labels
    
    elif args.dataset.startswith('cifar100-n'):
        _, noise_type = args.dataset.split("/")
        root_path = os.path.join(args.dataset_root,"cifarn")
        trainset = CIFAR100N(root=root_path, download=True, train=True, transform = None, noise_type = noise_type,
                             noise_path = os.path.join(root_path,'CIFAR-100_human.pt'), is_human = True)
        noisy_labels = trainset.train_noisy_labels
    
    return np.array(noisy_labels).astype(int)


def get_databases_medmnist(args):
    """
    Load pre-generated MedMNIST embeddings and labels from stored .npz files.
    Args:
        args (Namespace): Parsed command line arguments.
    Returns:
        tuple: Embeddings and labels for training, validation, and testing.
    """

    db_path = os.path.join(args.database_root, 
                           args.dataset, 
                           args.backbone)    
    
    train_path = os.path.join(db_path,'train.npz')
    train_data = np.load(train_path)

    x_train = train_data['embeddings']
    y_train = train_data['labels'].astype(int).flatten()

    val_path = os.path.join(db_path,'val.npz')
    val_data = np.load(val_path)

    x_val = val_data['embeddings']
    y_val = val_data['labels'].astype(int).flatten()

    test_path = os.path.join(db_path,'test.npz')
    test_data = np.load(test_path)

    x_test = test_data['embeddings']
    y_test = test_data['labels'].astype(int).flatten()

    return x_train, y_train, x_val, y_val, x_test, y_test


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Dataset loading functions

def get_cifar10(args, transforms):
    data_path = os.path.join(args.dataset_root,args.dataset)
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms)
    return trainset, testset 


def get_cifar100(args, transforms):
    data_path = os.path.join(args.dataset_root,args.dataset)
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms)
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms)
    return trainset, testset 


def get_cifar10n(args, transforms):
    # e.g. dataset-name = cifar10-n/aggre_label
    #           split it into dataset, noise : cifar10-n, aggre_label
    dataset, noise_type = args.dataset.split("/")
    root_dir = os.path.join(args.dataset_root,dataset)
    
    trainset = CIFAR10N(root=root_dir,
                        download=True,  
                        train=True, 
                        transform = transforms,
                        noise_type = noise_type,
                        noise_path = os.path.join(root_dir,'CIFAR-10_human.pt'), 
                        is_human = True
                )
    
    testset = CIFAR10N(root=root_dir,
                       download=False,  
                       train=False, 
                       transform = transforms,
                       noise_type=noise_type,
                )

    return trainset, testset 


def get_cifar100n(args, transforms):
    # e.g. dataset-name = cifar100-n/noisy_label
    #           split it into dataset, noise : cifar100-n, noisy_label
    dataset, noise_type = args.dataset.split("/")
    root_path = os.path.join(args.dataset_root,dataset)
    
    trainset = CIFAR100N(root=root_path,
                         download=True,  
                         train=True, 
                         transform = transforms,
                         noise_type = noise_type,
                         noise_path = os.path.join(root_path,'CIFAR-100_human.pt'), 
                         is_human = True
                )
    
    testset = CIFAR100N(root=root_path,
                        download=False,  
                        train=False, 
                        transform = transforms,
                        noise_type=noise_type,
                )

    return trainset, testset 


def get_mnist(args, transforms):
    # convert img to RGB & append after ToTensor()
    to_rgb = T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
    transforms.transforms.insert(-1, to_rgb)
    data_path = os.path.join(args.dataset_root,args.dataset)
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms)
    testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms)
    return trainset, testset 


def get_fmnist(args, transforms):
    # convert img to RGB & append after ToTensor()
    to_rgb = T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
    transforms.transforms.insert(-1, to_rgb)
    data_path = os.path.join(args.dataset_root,args.dataset)
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms)
    testset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms)
    return trainset, testset


def get_animal10n(args, transforms):
    data_path = os.path.join(args.dataset_root,args.dataset)
    trainset = ANIMAL10N(img_dir=data_path + '/training', transform=transforms)
    testset = ANIMAL10N(img_dir=data_path + '/testing', transform=transforms)
    return trainset, testset 