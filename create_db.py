from src.utils.utils import init_db_folders, seed_everything
from src.utils.data import get_dataloaders, extract_embeddings

from argparse import ArgumentParser

import numpy as np
import os

import timm
import torch


def main(args):
    """
    Main function to extract embeddings using a pre-trained model and save them as databases.
    Args:
        args (Namespace): Parsed command line arguments.
    """
    
    # Set seeds for reproducibility
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Define the device to be used (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model from timm with specified backbone
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0).to(device)
    model.requires_grad_(False)  # Freeze the model parameters
    model = model.eval()  # Set the model to evaluation mode

    # Resolve model-specific data configuration for preprocessing
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Prepare data loaders for training and testing
    dataloaders = get_dataloaders(args, transforms, g)

    # Create necessary database folders if they do not exist
    init_db_folders(args)

    for split in ['train', 'test']:
        # Extract embeddings from the model for the given dataset split
        db = extract_embeddings(
            model=model,
            device=device,
            dataloader=dataloaders[split]
        )

        # Save the extracted embeddings and labels to disk as an .npz file
        np.savez(
            os.path.join(args.database_root, args.dataset, args.backbone, f'{split}.npz'),
            **db
        )


if __name__ == '__main__':
    parser = ArgumentParser(description="Extract and save embeddings using a pre-trained model.")

    # General settings
    parser.add_argument('--dataset_root', type=str, default="assets/data",
                        help='Path to the root directory of the dataset.')
    parser.add_argument('--database_root', type=str, default="assets/database",
                        help='Path to the root directory where the embeddings will be stored.')

    # Dataset and Model Configuration
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process (required).')
    parser.add_argument('--backbone', type=str, default='vit_large_patch14_dinov2.lvd142m',
                        help='Feature extractor backbone model from timm (default: vit_large_patch14_dinov2.lvd142m).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data loading (default: 128).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)
