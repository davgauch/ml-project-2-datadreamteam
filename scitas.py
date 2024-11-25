import pandas as pd
import numpy as np 

import torch

from dataset import WebcamDataset
from model import CNN_LSTM
from trainer import Trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

BATCH_SIZE = 32

def main(DATA_FOLDER):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading datasets...")
    train_dataset = WebcamDataset(images_path=f"{DATA_FOLDER}/X_BC_train.npy", ghi_values_path=f"{DATA_FOLDER}/ground_truth_train.npy")
    val_dataset = WebcamDataset(images_path=f"{DATA_FOLDER}/X_BC_val.npy", ghi_values_path=f"{DATA_FOLDER}/ground_truth_val.npy")
    test_dataset = WebcamDataset(images_path=f"{DATA_FOLDER}/X_BC_test.npy", ghi_values_path=f"{DATA_FOLDER}/ground_truth_test.npy")
    
    print("Creating the model...")
    # Instantiate the model
    model = CNN_LSTM()

    print("Converting the model...")

    bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }

    dnn_to_bnn(model, bnn_prior_parameters)

    # Create the Trainer object
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE, epochs=10, device=device, model_save_folder="output/model/", train_loss_file="output/train_losses.csv", test_loss_file="output/test_losses.csv")

    print(f"Training the model (device: {device}) ...")

    # Start training
    trainer.train()

    print("\nEvaluating the model...")

    # Evaluate on the test set
    trainer.evaluate()

if __name__ == '__main__':
    main("./data")