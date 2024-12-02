import csv
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from early_stopper import EarlyStopper
from interval_prediction import predict_intervals, display_prediction_intervals
import numpy as np

# Define Quantile Loss for quantile regression
class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.25, 0.75]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, y_pred, y_true):
        loss = 0.0
        for i, quantile in enumerate(self.quantiles):
            error = y_true[:, i] - y_pred[:, i]  # Directly index for each quantile
            loss += torch.mean(torch.max((quantile - 1) * error, quantile * error))  # Quantile loss
        return loss

# RMSE loss for standard regression
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

# Trainer class that supports both RMSE and Quantile Loss
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, gpu_id, 
                 quantile_reg=False, batch_size=32, learning_rate=1e-3, epochs=10, 
                 save_every=1, model_snapshot_file="output/model_snapshot.pt", 
                 train_loss_file="output/train_losses.csv", test_loss_file="output/test_losses.csv", 
                 quantiles=[0.25, 0.75]):
        
        self.gpu_id = gpu_id
        self.model = model

        # Choose loss function based on quantile_reg flag
        self.criterion = QuantileLoss(quantiles) if quantile_reg else RMSELoss()

        # Optimizer and scheduler setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)

        # Other settings
        self.epochs = epochs
        self.save_every = save_every
        self.model_snapshot_file = model_snapshot_file
        self.train_loss_file = train_loss_file
        self.test_loss_file = test_loss_file
        self.early_stopper = EarlyStopper(patience=15, min_delta=0.02)

        # Prepare data loaders
        self._prepare_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

        # Ensure directories exist for saving files
        self._ensure_directories_exist()

        # Initialize loss file for logging
        self._initialize_loss_file()

    def _prepare_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size):
        """Prepare the data loaders for training, validation, and testing."""
        if torch.cuda.is_available():
            self.model = self.model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset)
            test_sampler = DistributedSampler(test_dataset)
            pin_memory = True
        else:
            train_sampler = val_sampler = test_sampler = None
            pin_memory = False

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                       pin_memory=pin_memory, sampler=train_sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                     pin_memory=pin_memory, sampler=val_sampler)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                      pin_memory=pin_memory, sampler=test_sampler)

    def _ensure_directories_exist(self):
        """Ensure that directories for saving files exist."""
        os.makedirs(os.path.dirname(self.model_snapshot_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.train_loss_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.test_loss_file), exist_ok=True)

    def _initialize_loss_file(self):
        """Initialize the training loss CSV file."""
        if self.train_loss_file:
            with open(self.train_loss_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Processed Train Batches', 
                                 'Skipped Train Batches', 'Processed Val Batches', 'Skipped Val Batches', 
                                 'Train Time (s)', 'Eval Time (s)'])

    def _expand_labels_for_quantiles(self, labels, num_quantiles):
        """Expand labels for quantile regression (only if needed)."""
        return labels.view(-1, 1).expand(-1, num_quantiles)

    def train(self):
        """Main function to train the model."""
        for epoch in range(self.epochs):
            start_train_time = time.time()
            train_loss, processed_train_batches, skipped_train_batches = self.train_one_epoch(epoch)
            train_time = time.time() - start_train_time

            print(f"Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.4f}, "
                  f"Processed Train Batches: {processed_train_batches}, "
                  f"Skipped Train Batches: {skipped_train_batches}, Train Time: {train_time:.2f}s")
            
            start_eval_time = time.time()
            val_loss, processed_val_batches, skipped_val_batches = self.evaluate()
            eval_time = time.time() - start_eval_time

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {val_loss:.4f}, "
                  f"Processed Validation Batches: {processed_val_batches}, "
                  f"Skipped Validation Batches: {skipped_val_batches}, Eval Time: {eval_time:.2f}s")
            
            epoch_results = [epoch, train_loss, val_loss, processed_train_batches, skipped_train_batches, 
                             processed_val_batches, skipped_val_batches, train_time, eval_time]
            
            if self.train_loss_file and self.gpu_id == 0:
                with open(self.train_loss_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(epoch_results)

            if self.model_snapshot_file and self.gpu_id == 0 and epoch % self.save_every == 0:
                snapshot = {"MODEL_STATE": self.model.state_dict(), "EPOCHS_RUN": epoch}
                torch.save(snapshot, self.model_snapshot_file)
                print(f"Epoch {epoch} | Training snapshot saved at {self.model_snapshot_file}")

            if self.early_stopper.early_stop(val_loss):
                print("Early stopped training.")
                break

        print("Training complete. Losses and batch info saved to:", self.train_loss_file)

    def train_one_epoch(self, epoch):
        """Performs one epoch of training."""
        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        
        if torch.cuda.is_available():
            self.train_loader.sampler.set_epoch(epoch)

        self.model.train()

        running_loss = 0.0
        processed_batches = 0
        skipped_batches = 0

        for inputs, labels in self.train_loader:
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                skipped_batches += 1
                print("Warning: Skipping batch due to NaN values.")
                continue

            if torch.cuda.is_available():
                inputs, labels = inputs.to(self.gpu_id).float(), labels.to(self.gpu_id).float()
            else:
                inputs, labels = inputs.to("cpu").float(), labels.to("cpu").float()

            # Expand labels if quantile regression is being used
            if isinstance(self.criterion, QuantileLoss):
                labels = self._expand_labels_for_quantiles(labels, len(self.criterion.quantiles))

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Handle tuple outputs (e.g., from Monte Carlo Dropout)
            if isinstance(outputs, tuple):
                mean_prediction, lower_bound, upper_bound = outputs
                outputs = mean_prediction


            if torch.isnan(outputs).any():
                skipped_batches += 1
                print("Warning: Skipping batch due to NaN in outputs.")
                continue

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            processed_batches += 1

        avg_loss = running_loss / max(1, processed_batches)
        return avg_loss, processed_batches, skipped_batches


    def evaluate(self):
        """
        Evaluates the model on the validation dataset, calculates the mean predictions, and 95% confidence intervals.
        """
        self.model.eval()
        running_loss = 0.0
        processed_batches = 0
        skipped_batches = 0

        predictions = []
        lower_bounds = []
        upper_bounds = []
        ground_truths = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                # Skip batches with NaN values
                if torch.isnan(inputs).any() or torch.isnan(labels).any():
                    skipped_batches += 1
                    print("Warning: Skipping batch due to NaN values.")
                    continue

                # Move inputs and labels to the appropriate device
                if torch.cuda.is_available():
                    inputs, labels = inputs.to(self.gpu_id).float(), labels.to(self.gpu_id).float()
                else:
                    inputs, labels = inputs.to("cpu").float(), labels.to("cpu").float()

                # Get predictions and uncertainties
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):  # For Monte Carlo Dropout models
                    mean_prediction, lower_bound, upper_bound = outputs
                else:
                    raise ValueError("The model must return a tuple of (mean, lower_bound, upper_bound).")

                # Save results for analysis
                predictions.append(mean_prediction.cpu())
                lower_bounds.append(lower_bound.cpu())
                upper_bounds.append(upper_bound.cpu())
                ground_truths.append(labels.cpu())

                # Check for NaN in outputs
                if torch.isnan(mean_prediction).any():
                    skipped_batches += 1
                    print("Warning: Skipping batch due to NaN in outputs.")
                    continue

                # Compute loss
                loss = self.criterion(mean_prediction, labels)
                running_loss += loss.item()
                processed_batches += 1

        # Aggregate results
        predictions = torch.cat(predictions).numpy()
        lower_bounds = torch.cat(lower_bounds).numpy()
        upper_bounds = torch.cat(upper_bounds).numpy()
        ground_truths = torch.cat(ground_truths).numpy()

        # Save results to a file
        results = {
            "predictions": predictions,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "ground_truths": ground_truths,
        }
        output_path = "mc_dropout_results.npy"
        np.save(output_path, results)
        print(f"Evaluation results saved to {output_path}")

        # Compute average loss
        avg_loss = running_loss / max(1, processed_batches)

        return avg_loss, processed_batches, skipped_batches
