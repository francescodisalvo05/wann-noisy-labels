from src.utils.utils import seed_everything, get_run_id_knn
from src.utils.preprocessing import preprocessing_knn
from src.utils.data import get_databases

from src.model.ANN import ANN
from src.model.WANN import WANN
from src.model.KNN import KNN

from argparse import ArgumentParser

import numpy as np
import json
import os


def get_knn_model(args):
    """
    Retrieve the appropriate k-NN model based on the method specified in args.
    Args:
        args (Namespace): Parsed command line arguments containing model configuration.
    Returns:
        model: An instance of KNN, ANN, or WANN depending on the chosen method.
    """
    if args.method == "kNN":
        model = KNN(k=args.k)
    elif args.method == "ANN":
        model = ANN(kmin=args.kmin, kmax=args.kmax)
    elif args.method == "WANN":
        model = WANN(kmin=args.kmin, kmax=args.kmax)
    return model


def main(args):
    """
    Main function to train and evaluate k-NN based models (KNN, ANN, WANN).
    Args:
        args (Namespace): Parsed command line arguments.
    """
    
    # Set random seed for reproducibility
    seed_everything(args.seed)

    # Load and preprocess datasets
    x_train, y_train, x_test, y_test = get_databases(args)
    x_train, y_train, x_test, y_test = preprocessing_knn(args, x_train, y_train, x_test, y_test)

    # Initialize the selected model
    model = get_knn_model(args)

    # Train the model
    model.fit(x_train, y_train)

    # Predict using the trained model
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = np.sum(y_pred == y_test) / len(y_test) * 100
    print(f"[{args.method}] Accuracy = {accuracy:.2f}")    

    # Prepare logs for saving
    k_info = args.k if args.method == "kNN" else (args.kmin, args.kmax) 
    logs = {
        'dataset': args.dataset,
        'backbone': args.backbone,
        'method': args.method,
        'k': k_info,
        'accuracy': accuracy,
        'artificial_noise': {
            'pattern': args.noise_pattern,
            'rate': args.noise_ratio
        },
        'imbalance_ratio': args.imbalance_ratio,
        'dimensionality_reduction': args.dimensionality_reduction
    }

    # Create directory for experiment logs
    experiment_path = os.path.join(args.output_root, args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    # Save logs as a JSON file
    filepath = os.path.join(experiment_path, f'{get_run_id_knn(args)}.json')
    print(filepath)
    with open(filepath, 'w') as f:
        json.dump(logs, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()

    # General settings
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Define the name of the experiment.')
    parser.add_argument('--dataset_root', type=str, default="assets/data",
                        help='Path to the root directory of the dataset.')
    parser.add_argument('--database_root', type=str, default="assets/database",
                        help='Path to the root directory of the database.')
    parser.add_argument('--output_root', type=str, default='assets/logs/',
                        help='Path to the root directory where logs will be stored.')
    parser.add_argument('--backbone', type=str, default='vit_large_patch14_dinov2.lvd142m',
                        help='Define the backbone model.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    # Dataset & Hyperparameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process (required).')
    parser.add_argument('--method', type=str, default='WANN',
                        choices=["WANN", "ANN", "kNN"],
                        help='Inference method to use. Choices: "WANN", "ANN", "kNN". Default: "WANN".')
    parser.add_argument('--k', type=int, default=11,
                        help='Set k if method == "kNN". Default: 11.')
    parser.add_argument('--kmin', type=int, default=11,
                        help='Set kmin if method == "WANN" or "ANN". Default: 11.')
    parser.add_argument('--kmax', type=int, default=51,
                        help='Set kmax if method == "WANN" or "ANN". Default: 51.')

    # Noise Handling
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='Define the noise ratio (default: 0.0).')
    parser.add_argument('--noise_pattern', type=str, default="None",
                        choices=["sym", "asym", "instance", "None"],
                        help='Define the noise pattern. Choices: "sym", "asym", "instance", "None".')
    parser.add_argument('--num_subsamples', type=int, default=-1,
                        help="Define the total number of desired samples (train+val). Default: -1 (use all).")

    # Imbalance Handling
    parser.add_argument('--imbalance_ratio', type=float, default=None,
                        help="Define the imbalance ratio (default: None).")

    # Dimensionality Reduction
    parser.add_argument('--dimensionality_reduction', type=str, default="None",
                        choices=["None", "PCA", "LDA", "fLDA"],
                        help="Define the dimensionality reduction method. Choices: 'None', 'PCA', 'LDA', 'fLDA'.")

    args = parser.parse_args()

    # Run the main function
    main(args)