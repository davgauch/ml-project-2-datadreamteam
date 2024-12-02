import pandas as pd
import numpy as np 

import torch

from dataset import WebcamDataset
from model import CNN_LSTM
from quantile_regression import QuantileRegressionModel
from monte_carlo_dropout import MonteCarloDropoutModel
from trainer import Trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import argparse

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
   
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
   

def main(rank, world_size, epochs, save_every, data_dir, working_dir, batch_size, bayesian, quantile, monte_carlo_dropout, mc_samples):
    print("Setuping DDP...", flush=True)
    if world_size > 1:
        ddp_setup(rank=rank, world_size=world_size)

    print("Loading datasets...", flush=True)
    # Don't use subset in case of multi gpu training
    train_dataset = WebcamDataset(images_path=f"{data_dir}/normalized_X_BC_train.npy", ghi_values_path=f"{data_dir}/normalized_labels_train.npy")
    val_dataset = WebcamDataset(images_path=f"{data_dir}/normalized_X_BC_val.npy", ghi_values_path=f"{data_dir}/normalized_labels_val.npy")
    test_dataset = WebcamDataset(images_path=f"{data_dir}/normalized_X_BC_test.npy", ghi_values_path=f"{data_dir}/normalized_labels_test.npy")
    
    print("Creating the model...", flush=True)
    # Instantiate the model
    model = CNN_LSTM()
    print("Model before any transformation:", model)
    # Handle Bayesian option
    if bayesian:
        bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Reparameterization",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.5,
        }
        print("Converting the model to Bayesian...", flush=True)
        dnn_to_bnn(model, bnn_prior_parameters)
        print("Model after transformation:", model)

    # Handle Quantile Regression option
    if quantile:
        print("Setting up model for Quantile Regression...", flush=True)
        print("Model before quantile regression:", model)
        model = QuantileRegressionModel(model)  
        print("Model after quantile regression:", model)

    # Handle Monte Carlo Dropout option
    if monte_carlo_dropout:  # NEW: Check if Monte Carlo Dropout is enabled
        print("Wrapping model for Monte Carlo Dropout...", flush=True)
        model = MonteCarloDropoutModel(model, num_samples=mc_samples)

    # Convert BatchNorm to sync (needed for sync between GPUs)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print("Creating the trainer object...", flush=True)

    # Create the Trainer object
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, gpu_id=rank, batch_size=batch_size, epochs=epochs, model_snapshot_file=f"{working_dir}/model.pt", train_loss_file=f"{working_dir}/train_losses.csv", test_loss_file=f"{working_dir}/test_losses.csv", quantile_reg = quantile)

    print(f"Training the model ...", flush=True)

    # Start training
    trainer.train()

    if monte_carlo_dropout:
        print("Evaluating with Monte Carlo Dropout...", flush=True)
        evaluate_monte_carlo_dropout(model, test_dataset, batch_size, rank)

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
    all_uncertainties = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.cuda(rank)
            outputs = model(x)

            if len(outputs) == 3:
                mean_prediction, lower_bound, upper_bound = outputs
                uncertainty = upper_bound - lower_bound  # Width of the confidence interval
            elif len(outputs) == 2:
                mean_prediction, uncertainty = outputs
            else:
                raise ValueError(f"Unexpected number of outputs from model: {len(outputs)}")

            all_mean_predictions.append(mean_prediction.cpu())
            all_uncertainties.append(uncertainty.cpu())

    print("Mean Predictions:", torch.cat(all_mean_predictions))
    print("Uncertainties:", torch.cat(all_uncertainties))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('data_dir', type=str, help='Directory to find the data')
    parser.add_argument('working_dir', type=str, help='Folder where the output file and the saved items will go')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--quantile', action='store_true', help='Enable Quantile Regression')
    parser.add_argument('--bayesian', action='store_true', help='Enable Bayesian Conversion')
    parser.add_argument('--monte_carlo_dropout', action='store_true', help='Enable Monte Carlo Dropout')
    parser.add_argument('--mc_samples', default=50, type=int, help='Number of stochastic forward passes for Monte Carlo Dropout (default: 50)')

    args = parser.parse_args()

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()  
        print(f"Cuda available! Working with {world_size} GPUs")
    
        mp.spawn(main, args=(world_size, args.total_epochs, args.save_every, args.data_dir, args.working_dir, args.batch_size, args.bayesian, args.quantile, args.monte_carlo_dropout, args.mc_samples), nprocs=world_size)
    else:
        print(f"Cuda not available! Working with single threaded CPU.")
        main(rank=0, world_size=1, epochs=args.total_epochs, save_every=args.save_every, data_dir=args.data_dir, working_dir=args.working_dir, batch_size=args.batch_size, bayesian=args.bayesian, quantile = args.quantile, monte_carlo_dropout=args.monte_carlo_dropout, mc_samples=args.mc_samples)
