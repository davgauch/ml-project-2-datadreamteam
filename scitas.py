import pandas as pd
import numpy as np
import torch
from dataset import WebcamDataset
from model import CNN_LSTM
from monte_carlo_dropout import MonteCarloDropoutModel
from trainer import Trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import argparse


def ddp_setup(rank: int, world_size: int):
    """
    Sets up distributed data parallel (DDP).
    Args:
        rank: Unique identifier of each process.
        world_size: Total number of processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size, epochs, save_every, data_dir, working_dir, batch_size, bayesian, monte_carlo_dropout, mc_samples, normalized, skip_training, model_path, subset):
    print("Setting up DDP...", flush=True)
    if world_size > 1:
        ddp_setup(rank=rank, world_size=world_size)

    print("Loading datasets...", flush=True)
    train_dataset = WebcamDataset(images_path=f"{data_dir}/normalized_X_BC_train.npy", ghi_values_path=f"{data_dir}/normalized_labels_train.npy")
    val_dataset = WebcamDataset(images_path=f"{data_dir}/normalized_X_BC_val.npy", ghi_values_path=f"{data_dir}/normalized_labels_val.npy")
    test_dataset = WebcamDataset(images_path=f"{data_dir}/normalized_X_BC_test.npy", ghi_values_path=f"{data_dir}/normalized_labels_test.npy")

    print("Creating the model...", flush=True)
    model = CNN_LSTM()

    if bayesian:
        bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Reparameterization",
            "moped_enable": False,
            "moped_delta": 0.5,
        }
        print("Converting the model to Bayesian...", flush=True)
        dnn_to_bnn(model, bnn_prior_parameters)

    if monte_carlo_dropout:
        print("Wrapping model for Monte Carlo Dropout...", flush=True)
        model = MonteCarloDropoutModel(model, mc_samples=mc_samples)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print("Creating the trainer object...", flush=True)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        gpu_id=rank,
        batch_size=batch_size,
        learning_rate=1e-3,
        epochs=epochs,
        save_every=save_every,
        working_dir=working_dir,
        mc_samples=mc_samples,
        model_snapshot_file=f"{working_dir}/model.pt",
        train_loss_file=f"{working_dir}/train_losses.csv",
        test_loss_file=f"{working_dir}/test_losses.csv"
    )

    if not skip_training:
        print(f"Training the model...", flush=True)
        trainer.train()

    if monte_carlo_dropout:
        print("Evaluating with Monte Carlo Dropout...", flush=True)
        evaluate_monte_carlo_dropout(model, test_dataset, batch_size, rank)

    if world_size > 1:
        print("\nDestroying the process group...", flush=True)
        destroy_process_group()


def evaluate_monte_carlo_dropout(model, test_dataset, batch_size, rank):
    """
    Evaluate the model with Monte Carlo Dropout enabled.
    Args:
        model (MonteCarloDropoutModel): The model wrapped for MC Dropout.
        test_dataset: Dataset for testing.
        batch_size: Batch size for evaluation.
        rank: GPU device ID.
    """
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model.eval()
    all_mean_predictions = []
    all_lower_bounds = []
    all_upper_bounds = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.cuda(rank)
            mean_prediction, lower_bound, upper_bound = model(x)
            all_mean_predictions.append(mean_prediction.cpu())
            all_lower_bounds.append(lower_bound.cpu())
            all_upper_bounds.append(upper_bound.cpu())

    mean_predictions = torch.cat(all_mean_predictions)
    lower_bounds = torch.cat(all_lower_bounds)
    upper_bounds = torch.cat(all_upper_bounds)

    print("Mean Predictions:", mean_predictions)
    print("Lower Bounds:", lower_bounds)
    print("Upper Bounds:", upper_bounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed training with Monte Carlo Dropout.')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='Frequency (in epochs) to save the model')
    parser.add_argument('data_dir', type=str, help='Directory containing the data')
    parser.add_argument('working_dir', type=str, help='Directory to save outputs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default: 32)')
    parser.add_argument('--bayesian', action='store_true', help='Enable Bayesian transformation')
    parser.add_argument('--monte_carlo_dropout', action='store_true', help='Enable Monte Carlo Dropout')
    parser.add_argument('--mc_samples', default=50, type=int, help='Number of stochastic forward passes for MC Dropout (default: 50)')
    parser.add_argument('--normalized', default=False, type=bool, help='Use normalized datasets')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and evaluate the model directly')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model checkpoint')
    parser.add_argument('--subset', type=int, default=None, help='Subset size for debugging (default: None)')

    args = parser.parse_args()

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"CUDA available! Using {world_size} GPUs.", flush=True)
        mp.spawn(
            main,
            args=(world_size, args.total_epochs, args.save_every, args.data_dir, args.working_dir, args.batch_size, args.bayesian, args.monte_carlo_dropout, args.mc_samples, args.normalized, args.skip_training, args.model_path, args.subset),
            nprocs=world_size
        )
    else:
        print("CUDA not available! Using CPU.", flush=True)
        main(rank=0, world_size=1, epochs=args.total_epochs, save_every=args.save_every, data_dir=args.data_dir, working_dir=args.working_dir, batch_size=args.batch_size, bayesian=args.bayesian, monte_carlo_dropout=args.monte_carlo_dropout, mc_samples=args.mc_samples, normalized=args.normalized, skip_training=args.skip_training, model_path=args.model_path, subset=args.subset)
