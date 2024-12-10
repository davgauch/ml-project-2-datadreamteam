import csv
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.stats import norm
import numpy as np
from early_stopper import EarlyStopper


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, gpu_id, 
                 batch_size=32, learning_rate=1e-3, epochs=10, 
                 save_every=1, working_dir="/output", model_snapshot_file=None, 
                 train_loss_file=None, test_loss_file=None, mc_samples=50):
        """
        Initializes the Trainer class with all the necessary components.
        """
        # Store the epochs parameter
        self.epochs = epochs  # FIX: Store epochs as an instance attribute
        self.gpu_id = gpu_id
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_every = save_every
        self.mc_samples = mc_samples  # Store Monte Carlo samples

        self.model_snapshot_file = model_snapshot_file or f"{working_dir}/model.pt"
        self.train_loss_file = train_loss_file or f"{working_dir}/train_losses.csv"
        self.test_loss_file = test_loss_file or f"{working_dir}/test_losses.csv"

        self.test_preds_file = os.path.join(working_dir, "preds.npy")
        self.test_true_labels_file = os.path.join(working_dir, "true_labels.npy")
        self.test_pred_lower_bounds_file = os.path.join(working_dir, "pred_lower_bounds.npy")
        self.test_pred_upper_bounds_file = os.path.join(working_dir, "pred_upper_bounds.npy")

        # Loss function, optimizer, and scheduler
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        self.early_stopper = EarlyStopper(patience=20, min_delta=0.01)

        # Prepare datasets
        self.train_loader, self.val_loader, self.test_loader = self._prepare_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size
        )

        # Create output directories
        os.makedirs(working_dir, exist_ok=True)

        # Initialize the loss file
        with open(self.train_loss_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Train Time (s)', 'Eval Time (s)'])


    def _prepare_dataloaders(self, train_dataset, val_dataset, test_dataset, batch_size):
        """Prepares the dataloaders with optional distributed sampling."""
        if torch.cuda.is_available():
            self.model = DDP(self.model.to(self.gpu_id), device_ids=[self.gpu_id])
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
        else:
            train_sampler = val_sampler = test_sampler = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)

        return train_loader, val_loader, test_loader

    def train(self):
        """Trains the model over the specified number of epochs."""
        for epoch in range(self.epochs):
            start_train_time = time.time()
            train_loss = self._train_one_epoch(epoch)
            train_time = time.time() - start_train_time

            start_eval_time = time.time()
            val_loss = self._evaluate(self.val_loader)
            eval_time = time.time() - start_eval_time

            self.scheduler.step(val_loss)
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save metrics and model
            with open(self.train_loss_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, train_loss, val_loss, train_time, eval_time])

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

            if self.early_stopper.early_stop(val_loss):
                print("Early stopping triggered.")
                break

    def _train_one_epoch(self, epoch):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0
        for inputs, labels in self.train_loader:
            # Move data to the appropriate device
            inputs, labels = inputs.to(self.gpu_id), labels.to(self.gpu_id)

            self.optimizer.zero_grad()

            # Perform multiple forward passes for Monte Carlo Dropout
            mc_outputs = torch.stack([self.model(inputs)[0] for _ in range(self.mc_samples)])  # Use only the mean prediction
            mean_prediction = mc_outputs.mean(0)

            # Compute loss
            loss = self.criterion(mean_prediction, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _evaluate(self, loader):
        """Evaluates the model on a dataset."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, labels in loader:
                # Move data to the appropriate device
                inputs, labels = inputs.to(self.gpu_id), labels.to(self.gpu_id)

                # Perform multiple forward passes for Monte Carlo Dropout
                mc_outputs = torch.stack([self.model(inputs)[0] for _ in range(self.mc_samples)])  # Use only the mean prediction
                mean_prediction = mc_outputs.mean(0)

                # Compute loss
                loss = self.criterion(mean_prediction, labels)
                total_loss += loss.item()

        return total_loss / len(loader)

    def test(self):
        """Evaluates the model on the test dataset using Monte Carlo Dropout."""
        self.model.eval()
        predictions, lower_bounds, upper_bounds, ground_truths = [], [], [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.gpu_id), labels.to(self.gpu_id)
                mc_outputs = torch.stack([self.model(inputs) for _ in range(self.mc_samples)])
                mean_prediction = mc_outputs.mean(0)
                uncertainty = mc_outputs.std(0)

                z_score = norm.ppf(0.975)
                lower_bound = mean_prediction - z_score * uncertainty
                upper_bound = mean_prediction + z_score * uncertainty

                predictions.append(mean_prediction.cpu().numpy())
                lower_bounds.append(lower_bound.cpu().numpy())
                upper_bounds.append(upper_bound.cpu().numpy())
                ground_truths.append(labels.cpu().numpy())

        # Save results
        np.save(self.test_preds_file, np.concatenate(predictions))
        np.save(self.test_true_labels_file, np.concatenate(ground_truths))
        np.save(self.test_pred_lower_bounds_file, np.concatenate(lower_bounds))
        np.save(self.test_pred_upper_bounds_file, np.concatenate(upper_bounds))
        print("Test results saved.")

    def _save_checkpoint(self, epoch):
        """Saves the model and optimizer state."""
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.model_snapshot_file)
        print(f"Checkpoint saved for epoch {epoch}.")
