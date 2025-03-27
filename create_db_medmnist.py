from src.utils.data import seed_worker, extract_embeddings

from torch.utils.data import DataLoader
from medmnist import INFO

import numpy as np
import medmnist
import argparse
import random
import torch
import timm
import os
                   

def main(args):
    """
    Main function to prepare datasets, extract embeddings using a pre-trained model, and save them.
    Args:
        args (Namespace): Parsed command line arguments.
    """

    # Enable deterministic algorithms for reproducibility
    torch.use_deterministic_algorithms(True)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Extract dataset metadata
    info = INFO[args.dataset]
    args.task, args.in_channel, args.num_classes = info['task'], info['n_channels'], len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    # Initialize model from timm with specified backbone
    model = timm.create_model(args.backbone, pretrained=True, num_classes=args.num_classes)

    # Get preprocessing configuration and create transforms
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Define datasets and dataloaders
    train_dataset = DataClass(split='train', transform=transforms, download=False, as_rgb=True, size=224, root=args.dataset_root, mmap_mode='r')
    val_dataset = DataClass(split='val', transform=transforms, download=False, as_rgb=True, size=224, root=args.dataset_root, mmap_mode='r')
    test_dataset = DataClass(split='test', transform=transforms, download=False, as_rgb=True, size=224, root=args.dataset_root, mmap_mode='r')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    # Set model to evaluation mode and transfer to device
    model.eval()
    model = model.to(args.device)

    # Create necessary database directories
    db_dataset_path = os.path.join(args.database_root, args.dataset)
    db_dataset_backbone_path = os.path.join(db_dataset_path, args.backbone)
    os.makedirs(db_dataset_path, exist_ok=True)
    os.makedirs(db_dataset_backbone_path, exist_ok=True)

    # Extract and save embeddings for train, validation, and test sets
    for split, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
        embeddings = extract_embeddings(model, args.device, loader)
        np.savez(f'{db_dataset_backbone_path}/{split}.npz', **embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--dataset_root', type=str, default="assets/data/medmnist",
                        help='Path to the root directory of the MedMNIST dataset.')
    parser.add_argument('--database_root', type=str, default="assets/database",
                        help='Path to the root directory where the embeddings will be stored.')

    # Dataset and Model Configuration
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the MedMNIST dataset to process (required).')
    parser.add_argument('--backbone', type=str, default='vit_large_patch14_dinov2.lvd142m',
                        help='Feature extractor backbone model from timm (default: vit_large_patch14_dinov2.lvd142m).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data loading (default: 128).')

    args = parser.parse_args()    

    # Run the main function
    main(args)