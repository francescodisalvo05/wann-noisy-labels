from src.utils.preprocessing import preprocess_probing, preprocess_probing_medmnist
from src.utils.data import get_databases, get_databases_medmnist
from src.utils.utils import seed_everything, get_ckpt_path, get_run_id_probing
from src.model.litLinear import LinearModule

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import json
import os


def inference_model(args):
    """
    Run inference using a trained model and save the results.
    Args:
        args (Namespace): Parsed command line arguments.
    """
    
    # Load datasets and preprocess them appropriately
    if args.dataset in ['dermamnist','breastmnist']:
        x_train, y_train, x_val, y_val, x_test, y_test = get_databases_medmnist(args)
        _, _, test_loader = preprocess_probing_medmnist(args, x_train, y_train, x_val, y_val, x_test, y_test)
    else:
        x_trainval, y_trainval, x_test, y_test = get_databases(args)
        _, _, test_loader = preprocess_probing(args, x_trainval, y_trainval, x_test, y_test)
    

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=False
    )

    # Load the trained model
    ckpt = get_ckpt_path(args, inference=True)
    model = LinearModule.load_from_checkpoint(ckpt, loss_str=args.loss_function)
    
    # Perform inference
    trainer.test(model, test_loader)
    accuracy = model.test_accuracy * 100

    # Save inference logs
    logs = {
        'dataset' : args.dataset,
        'backbone' : args.backbone,
        'loss' : args.loss_function,
        'accuracy' : accuracy,
        'arificial_noise' : {
            'pattern' : args.noise_pattern,
            'rate' : args.noise_ratio
        }
    }


    args.dataset = args.dataset.replace("/","-") # handle noisy ones
    experiment_path = os.path.join(args.output_root,args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    filepath = os.path.join(experiment_path, f'{get_run_id_probing(args)}.json')
    with open(filepath, 'w') as f:
        json.dump(logs, f, indent=4)
    

if __name__ == "__main__":
    
    parser = ArgumentParser()

    # General settings
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Define the name of the experiment.')
    parser.add_argument('--database_root', type=str, default='assets/database',
                        help='Path to the root directory where the embeddings will be stored.')
    parser.add_argument('--dataset_root', type=str, default="assets/data",
                        help='Path to the root directory of the dataset.')
    parser.add_argument('--backbone', type=str, default='vit_large_patch14_dinov2.lvd142m',
                        help='Feature extractor backbone model from timm.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--dataset', type=str, default="cifar10-n/random_label1",
                        help='Name of the dataset to process.')
    parser.add_argument('--store_ckpt_path', type=str, default='assets/ckpt/',
                        help='Path to the directory where checkpoints are stored.')
    parser.add_argument('--loss_function', type=str, default='ce',
                        help='Loss function used during training.')
    parser.add_argument('--output_root', type=str, default='assets/logs/',
                        help='Path to the root directory where logs will be stored.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for inference.')

    # Noise settings
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='Define the noise ratio to apply to the dataset.')
    parser.add_argument('--noise_pattern', type=str, default="None", choices=["sym", "asym", "instance", "None"],
                        help='Pattern of noise applied to the dataset.')
    parser.add_argument('--samples_per_class', type=int, default=-1,
                        help='Number of samples per class to use.')
    parser.add_argument('--num_subsamples', type=int, default=-1,
                        help='Total number of subsamples for training and validation.')

    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    seed_everything(args.seed)

    inference_model(args)