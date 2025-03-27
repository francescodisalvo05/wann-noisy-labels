from src.utils.data import get_databases,  get_databases_medmnist
from src.utils.preprocessing import preprocess_probing, preprocess_probing_medmnist
from src.utils.utils import seed_everything, get_ckpt_path, get_run_id_probing

from src.model.litLinear import LinearModule
from src.model.utils import get_loss, get_config_file
from src.model.loss import ELR

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser

import pytorch_lightning as pl
import numpy as np
import torch
import os 


def train_model(args):
    """
    Train a linear model using PyTorch Lightning with support for various datasets and logging.
    Args:
        args (Namespace): Parsed command line arguments.
    """

    # Load data according to dataset type
    if args.dataset in ['dermamnist', 'breastmnist']:
        x_train, y_train, x_val, y_val, _, _ = get_databases_medmnist(args)
        train_loader, val_loader = preprocess_probing_medmnist(args, x_train, y_train, x_val, y_val)
        num_classes = np.max(y_train).item() + 1
        embedding_dimension = len(x_train[0])
    else:
        x_trainval, y_trainval, _, _ = get_databases(args)
        train_loader, val_loader = preprocess_probing(args, x_trainval, y_trainval)
        num_classes = np.max(y_trainval).item() + 1
        embedding_dimension = len(x_trainval[0])
    
    # Define model checkpointing
    ckpt_path = get_ckpt_path(args)
    pl.seed_everything(args.seed, workers=True)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path, filename="{epoch}")
    checkpoint_callback_best = ModelCheckpoint(dirpath=ckpt_path, monitor=args.metric_to_monitor, mode=args.metric_to_monitor_mode, filename="best")

    early_stopping = EarlyStopping(monitor=args.metric_to_monitor, mode=args.metric_to_monitor_mode, patience=args.patience)

    wandb_logger = WandbLogger(project=f'wann_{args.experiment_name}', 
                               name=f'{get_run_id_probing(args)}',
                               mode=args.wandb)

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=args.num_epochs,
                         logger=wandb_logger if args.wandb else None, callbacks=[checkpoint_callback, checkpoint_callback_best, early_stopping],
                         check_val_every_n_epoch=1)

    # Setup loss function
    if args.loss_function == "elr":
        config_loss = get_config_file(args.dataset, args.loss_function)
        criterion = ELR(num_examp=len(train_loader.dataset), num_classes=num_classes, beta=config_loss["beta"], lam=config_loss["lambda"])
    else:
        criterion = get_loss(args.dataset, args.loss_function, num_classes)

    # Setup model
    model = LinearModule(num_classes=num_classes, criterion=criterion, embedding_dimension=embedding_dimension,
                         lr=args.learning_rate, loss_str=args.loss_function, patience=args.patience,
                         metric_to_monitor=args.metric_to_monitor, metric_to_monitor_mode=args.metric_to_monitor_mode,
                         weight_decay=args.weight_decay)

    # Train model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":

    parser = ArgumentParser()

    # General settings
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Define the name of the experiment.')
    parser.add_argument('--database_root', type=str, default='assets/database',
                        help='Path to the root directory where the embeddings will be stored.')
    parser.add_argument('--dataset_root', type=str, default="assets/data",
                        help='Path to the root directory of the dataset.')
    parser.add_argument('--output_root', type=str, default='assets/logs',
                        help='Path to the root directory where logs will be stored.')
    parser.add_argument('--store_ckpt_path', type=str, default='assets/ckpt',
                        help='Path to the directory where checkpoints will be stored.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--backbone', type=str, default='vit_large_patch14_dinov2.lvd142m',
                        help='Feature extractor backbone model from timm.')
    parser.add_argument('--wandb', type=str, default="online",
                        choices=["online", "offline"],
                        help='Enable or disable logging with WandB (online or offline).')

    # Dataset & Metric
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Name of the dataset to process.')
    parser.add_argument('--metric_to_monitor', type=str, default='Val/Accuracy',
                        help='Metric to monitor during training.')
    parser.add_argument('--metric_to_monitor_mode', type=str, default='max',
                        help='Goal of the metric to monitor (min or max).')

    # Noise & Size
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='Define the noise ratio to apply to the dataset.')
    parser.add_argument('--noise_pattern', type=str, default="None",
                        choices=["sym", "asym", "instance", "None"],
                        help='Pattern of noise to apply to the dataset.')
    parser.add_argument('--samples_per_class', type=int, default=-1,
                        help='Number of samples per class to use.')
    parser.add_argument('--num_subsamples', type=int, default=-1,
                        help='Total number of samples to use for training and validation.')

    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for training.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs with no improvement before early stopping.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--loss_function', type=str, default='ce',
                        help='Loss function to use for training (e.g., "ce" or "elr").')

    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    seed_everything(args.seed)

    train_model(args)
