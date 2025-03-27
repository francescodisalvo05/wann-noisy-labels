import numpy as np
import random
import torch
import os


def seed_everything(seed=42):
    """
    Ensure reproducibility across various libraries and frameworks.
    Args:
        seed (int): Seed value for random number generators.
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_db_folders(args):
    """
    Create database folders if they do not exist.
    The folder structure is:
        database_root / dataset-name / backbone
    Args:
        args (Namespace): Parsed command line arguments.
    """

    # Create root folder if necessary
    os.makedirs(args.database_root, exist_ok=True)

    # Create dataset folder
    dataset_path = os.path.join(args.database_root, args.dataset)
    os.makedirs(dataset_path, exist_ok=True)

    # Create backbone folder
    dataset_backbone_path = os.path.join(dataset_path, args.backbone)
    os.makedirs(dataset_backbone_path, exist_ok=True)


def get_run_id_knn(args):
    """
    Generate a unique identifier for k-NN experiment logs.
    Args:
        args (Namespace): Parsed command line arguments.
    Returns:
        str: Generated run ID.
    """

    dataset = args.dataset.replace("/","-") # handle noisy ones
    method = f"{args.k}NN" if args.method == "kNN" else args.method # else, WANN or ANN

    if args.num_subsamples > 0:
        experiment_tag = f"_{args.num_subsamples}samples"
    elif args.dimensionality_reduction != "None":
        experiment_tag = f"_dr-{args.dimensionality_reduction}"
    elif args.imbalance_ratio:
        experiment_tag = f"_lt-{args.imbalance_ratio}"
    else:
        experiment_tag = ""

    if args.noise_pattern != "None":
        run_id = f'{dataset}{experiment_tag}_{method}_{args.noise_pattern}-{args.noise_ratio}_s{args.seed}'
    else:
        run_id = f'{dataset}{experiment_tag}_{method}_s{args.seed}'

    return run_id


def get_run_id_probing(args):
    """
    Generate a unique identifier for probing experiment logs.
    Args:
        args (Namespace): Parsed command line arguments.
    Returns:
        str: Generated run ID.
    """

    dataset = args.dataset.replace("/","-") # handle noisy ones
    method = args.loss_function

    if args.num_subsamples > 0:
        experiment_tag = f"_{args.num_subsamples}samples"
    else:
        experiment_tag = ""

    if args.noise_pattern != "None":
        run_id = f'{dataset}{experiment_tag}_{method}_{args.noise_pattern}-{args.noise_ratio}_s{args.seed}'
    else:
        run_id = f'{dataset}{experiment_tag}_{method}_s{args.seed}'

    return run_id


def get_ckpt_path(args, inference=False):
    """
    Generate the checkpoint path for saving/loading models.
    Args:
        args (Namespace): Parsed command line arguments.
        inference (bool): If True, returns the path to the best checkpoint file.
    Returns:
        str: Checkpoint path (folder or file).
    """

    # Differentiate the checkpoint / noise
    if args.noise_pattern == "None":
        noise_ckpt_info = f'{args.noise_pattern}'
    else:
        print(f"noise ratio = {args.noise_ratio}")
        noise_ckpt_info = f'{args.noise_pattern}_{args.noise_ratio}'

    # Number of samples
    if args.num_subsamples > 0:
        path = os.path.join(args.store_ckpt_path,args.experiment_name,f'{args.num_subsamples}_samples',args.loss_function,args.dataset,noise_ckpt_info,f'SEED_{args.seed}')
    else:
        path = os.path.join(args.store_ckpt_path,args.experiment_name,args.loss_function,args.dataset,noise_ckpt_info,f'SEED_{args.seed}')

    # Handle inference case    
    if inference:
        return os.path.join(path, 'best.ckpt')
    else:
        return path