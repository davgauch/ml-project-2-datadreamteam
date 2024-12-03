import pandas as pd
import numpy as np 

import torch

from dataset import WebcamDataset
from model import CNN_LSTM
from quantile_regression import QuantileRegressionModel
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
   

def main(rank, world_size, epochs, save_every, data_dir, working_dir, batch_size, normalized, skip_training, model_path, subset):
    print("Setuping DDP...", flush=True)
    if world_size > 1:
        ddp_setup(rank=rank, world_size=world_size)

    print("Loading datasets...", flush=True)
    # Add the "normalized_" prefix if normalized is True
    prefix = "normalized_" if normalized else ""
    train_dataset = WebcamDataset(images_path=f"{data_dir}/{prefix}X_BC_train.npy", ghi_values_path=f"{data_dir}/{prefix}labels_train.npy", subset=subset)
    val_dataset = WebcamDataset(images_path=f"{data_dir}/{prefix}X_BC_val.npy", ghi_values_path=f"{data_dir}/{prefix}labels_val.npy", subset=subset)
    test_dataset = WebcamDataset(images_path=f"{data_dir}/{prefix}X_BC_test.npy", ghi_values_path=f"{data_dir}/{prefix}labels_test.npy", subset=subset)
    print(prefix)
    print("Creating the model...", flush=True)
    # Instantiate the model
    model = CNN_LSTM()
    # Handle Quantile Regression option
    print("Setting up model for Quantile Regression...", flush=True)
    print("Model before quantile regression:", model)
    model = QuantileRegressionModel(model)  
    print("Model after quantile regression:", model)
        
    # Convert BatchNorm to sync (needed for sync between GPUs)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print("Creating the trainer object...", flush=True)

    # Create the Trainer object
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, gpu_id=rank, batch_size=batch_size, epochs=epochs, working_dir=working_dir, model_path=model_path)

    if not skip_training:        
        print(f"Training the model...", flush=True)
        trainer.train()
        
    print(f"Testing the model...", flush=True)
    trainer.test()

    if world_size > 1:
        print("\nDestroying the process group...", flush=True)
        destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('data_dir', type=str, help='Directory to find the data')
    parser.add_argument('working_dir', type=str, help='Folder where the output file and the saved items will go')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--normalized', default=False, type=bool, help='Boolean to decide if you want to use normalized datasets')
    parser.add_argument('--skip_training', default=False, help='Boolean to run the model in testing mode')
    parser.add_argument('--model_path', default=None, type=str, help='Path to the saved model (.pt) for testing or resuming training')
    parser.add_argument('--subset', default=None, type=int, help='The size of the subset. None if full dataset')


    args = parser.parse_args()

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()  
        print(f"Cuda available! Working with {world_size} GPUs")
    
        mp.spawn(main, args=(world_size, args.total_epochs, args.save_every, args.data_dir, args.working_dir, args.batch_size, args.normalized, args.skip_training, args.model_path, args.subset), nprocs=world_size)
    else:
        print(f"Cuda not available! Working with single threaded CPU.")
        main(rank=0, world_size=1, epochs=args.total_epochs, save_every=args.save_every, data_dir=args.data_dir, working_dir=args.working_dir, batch_size=args.batch_size, normalized=args.normalized, skip_training=args.skip_training, model_path=args.model_path, subset=args.subset)
