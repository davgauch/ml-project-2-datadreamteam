import csv
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from early_stopper import EarlyStopper
from interval_prediction import predict_intervals

# Define Quantile Loss for quantile regression
class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.025, 0.5, 0.975]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, y_pred, y_true):
        loss = 0.0
        for i, quantile in enumerate(self.quantiles):
            error = y_true[:, i] - y_pred[:, i]  # Directly index for each quantile
            loss += torch.mean(torch.max((quantile - 1) * error, quantile * error))  # Quantile loss
        return loss

# Trainer class that supports both RMSE and Quantile Loss
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, gpu_id, batch_size=32, learning_rate=1e-3, epochs=10, 
                 save_every=1, working_dir="/output", model_path=None):
        """
        Initializes the Trainer class with all the necessary components.
        
        Args:
            model (torch.nn.Module): The model to train.
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            test_dataset (Dataset): The test dataset.
            gpu_id (int): The ID of the GPU to use for training (if applicable).
            batch_size (int, optional): The batch size to use during training. Default is 32.
            learning_rate (float, optional): The learning rate for the optimizer. Default is 1e-3.
            epochs (int, optional): The number of epochs to train for. Default is 10.
            save_every (int, optional): The frequency (in epochs) at which the model is saved. Default is 1.
            working_dir (str, optional): The directory where output (like saved models or logs) will be stored. Default is "/output".
            num_mc (int, optional): The number of Monte Carlo runs during training. Default is 3.
            num_monte_carlo (int, optional): The number of Monte Carlo samples to be drawn for inference. Default is 20.
        """
        self.gpu_id = gpu_id
        self.model = model
        self.start_epoch = 0

        self.criterion = QuantileLoss()

        # Optimizer and scheduler setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.early_stopper = EarlyStopper(patience=20, min_decrease=0.005)
        if model_path:
            print(f"Loading model from {model_path}...", flush=True)
            snapshot = torch.load(model_path)
            
            loaded_elements = []
            
            # Load model state
            if "MODEL_STATE" in snapshot:
                new_state_dict = {k.replace("module.", ""): v for k, v in snapshot["MODEL_STATE"].items()}
                self.model.load_state_dict(new_state_dict)
                loaded_elements.append("MODEL_STATE")
            
            # Load optimizer state
            if "OPTIMIZER_STATE" in snapshot:
                self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
                loaded_elements.append("OPTIMIZER_STATE")
            
            # Load scheduler state
            if "SCHEDULER_STATE" in snapshot:
                self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
                loaded_elements.append("SCHEDULER_STATE")
            
            # Set starting epoch
            start_epoch = snapshot.get("EPOCHS_RUN", -1) + 1
            loaded_elements.append("EPOCHS_RUN")
            
            # Restore EarlyStopper state
            early_stopper_state = snapshot.get("EARLY_STOPPER_STATE", None)
            if early_stopper_state:
                self.early_stopper.patience = early_stopper_state.get("patience", self.early_stopper.patience)
                self.early_stopper.min_decrease = early_stopper_state.get("min_decrease", self.early_stopper.min_decrease)
                self.early_stopper.counter = early_stopper_state.get("counter", self.early_stopper.counter)
                self.early_stopper.min_validation_loss = early_stopper_state.get("min_validation_loss", self.early_stopper.min_validation_loss)
                self.early_stopper.early_stop = early_stopper_state.get("early_stop", self.early_stopper.early_stop)
                loaded_elements.append("EARLY_STOPPER_STATE")
            
            print(f"Model loaded from {model_path}.")
            print(f"Loaded elements: {', '.join(loaded_elements)}")
        else:
            print("No model path provided, starting training from scratch.")

        # Prepare data loaders
        self._prepare_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

        self.epochs = epochs
        self.save_every = save_every

        self.working_dir = working_dir
        self.train_loss_file = f"{working_dir}/train_losses.csv"
        self.test_loss_file = f"{working_dir}/test_loss.csv"
        self.model_snapshot_file = f"{working_dir}/weights.pt"
        self.test_true_labels_file = f"{working_dir}/true_labels.npy"
        self.test_pred_lower_bounds_file = f"{working_dir}/pred_lower_bounds.npy"
        self.test_pred_mean_file= f"{working_dir}/pred_mean.npy"
        self.test_pred_upper_bounds_file = f"{working_dir}/pred_upper_bounds.npy"

        # Ensure the directories for saving models and loss files exist
        if self.working_dir:
            os.makedirs(self.working_dir, exist_ok=True)

        # Create or overwrite the loss files with headers
        if self.train_loss_file:
            with open(self.train_loss_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Processed Train Batches', 'Skipped Train Batches', 'Processed Val Batches', 'Skipped Val Batches', 'Train Time (s)', 'Eval Time (s)'])

    def _prepare_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size):
        """Prepare the data loaders for training, validation, and testing."""
        if torch.cuda.is_available():
            self.model = self.model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            test_sampler = DistributedSampler(test_dataset,shuffle=False)
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


    def _expand_labels_for_quantiles(self, labels, num_quantiles):
        """Expand labels for quantile regression (only if needed)."""
        return labels.view(-1, 1).expand(-1, num_quantiles)

    def train(self):
        """Main function to train the model."""
        for epoch in range(self.start_epoch, self.epochs):
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

            print(f"Epoch {epoch}/{self.epochs}, Validation Loss: {val_loss:.4f}, "
                  f"Processed Validation Batches: {processed_val_batches}, "
                  f"Skipped Validation Batches: {skipped_val_batches}, Eval Time: {eval_time:.2f}s")
            
            # Store losses and batch info for this epoch
            epoch_results = [epoch, train_loss, val_loss, processed_train_batches, skipped_train_batches, processed_val_batches, skipped_val_batches, train_time, eval_time]

            # Immediately write results to the CSV file after each epoch
            if self.train_loss_file and self.gpu_id == 0:
                with open(self.train_loss_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(epoch_results)

            # Saving the model after each epoch
            if self.model_snapshot_file and self.gpu_id == 0 and epoch % self.save_every == 0:
                snapshot = {
                    "MODEL_STATE": self.model.state_dict(),
                    "OPTIMIZER_STATE": self.optimizer.state_dict(),
                    "SCHEDULER_STATE": self.scheduler.state_dict(),
                    "EPOCHS_RUN": epoch,
                    "EARLY_STOPPER_STATE": {
                        "patience": self.early_stopper.patience,
                        "min_decrease": self.early_stopper.min_decrease,
                        "counter": self.early_stopper.counter,
                        "early_stop": self.early_stopper.early_stop,
                    },
                }
                torch.save(snapshot, self.model_snapshot_file)
                print(f"Epoch {epoch} | Training snapshot saved at {self.model_snapshot_file}")

            if self.early_stopper.early_stop(val_loss):             
                print("Early stopped training.")
                break

        print("Training complete. Losses and batch info saved to:", self.train_loss_file)

    def train_one_epoch(self, epoch):
        """Performs one epoch of training for paired image data."""
        b_sz = len(next(iter(self.train_loader))[0][0])  # Assuming train_loader yields a tuple of ((img1, img2), labels)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        
        if torch.cuda.is_available():
            self.train_loader.sampler.set_epoch(epoch)

        self.model.train()

        running_loss = 0.0
        processed_batches = 0
        skipped_batches = 0

        for (img1, img2, meteo_data), labels in self.train_loader:
            # NaN checks
            if torch.isnan(img1).any() or torch.isnan(img2).any() or torch.isnan(meteo_data).any() or torch.isnan(labels).any():
                skipped_batches += 1
                print("Warning: Skipping batch due to NaN values.")
                continue

            labels = self._expand_labels_for_quantiles(labels, len(self.criterion.quantiles))

            device = self.gpu_id if torch.cuda.is_available() else "cpu"
            img1, img2, meteo_data, labels = img1.to(device).float(), img2.to(device).float(), meteo_data.to(device).float(), labels.to(device).float()

            self.optimizer.zero_grad()

            # Forward pass with meteo_data
            outputs = self.model(img1, img2, meteo_data=meteo_data)

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
        Evaluates the model on the validation dataset.
        """
        self.model.eval()
        running_loss = 0.0
        processed_batches = 0
        skipped_batches = 0
    
        with torch.no_grad():
            for (img1, img2, meteo_data), labels in self.val_loader:
                if torch.isnan(img1).any() or torch.isnan(img2).any() or torch.isnan(meteo_data).any() or torch.isnan(labels).any():
                    skipped_batches += 1
                    print("Warning: Skipping batch due to NaN values.")
                    continue

                labels = self._expand_labels_for_quantiles(labels, len(self.criterion.quantiles))
                device = self.gpu_id if torch.cuda.is_available() else "cpu"
                img1, img2, meteo_data, labels = img1.to(device).float(), img2.to(device).float(), meteo_data.to(device).float(), labels.to(device).float()

                outputs = self.model(img1, img2, meteo_data=meteo_data)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                processed_batches += 1

        avg_loss = running_loss / max(1, processed_batches)
    
        return avg_loss, processed_batches, skipped_batches

        
    def test(self):
        """
        Evaluates the model on the test dataset using quantile regression.
        """
        self.model.eval()
        running_loss = 0.0
        processed_batches = 0
        skipped_batches = 0

        start_test_time = time.time()

        pred_intervals = []  # Collect prediction intervals
        true_labels = []     # Collect true labels (original, not expanded)

        with torch.no_grad():
            for (img1, img2, meteo_data), labels in self.test_loader:
                if torch.isnan(img1).any() or torch.isnan(img2).any() or torch.isnan(meteo_data).any() or torch.isnan(labels).any():
                    skipped_batches += 1
                    print("Warning: Skipping batch due to NaN values.")
                    continue

                original_labels = labels.cpu().numpy()
                true_labels.append(original_labels)

                labels = self._expand_labels_for_quantiles(labels, len(self.criterion.quantiles))
                device = self.gpu_id if torch.cuda.is_available() else "cpu"
                img1, img2, meteo_data, labels = img1.to(device).float(), img2.to(device).float(), meteo_data.to(device).float(), labels.to(device).float()

                # Use predict_intervals and model call with meteo_data
                lower_bound, mean, upper_bound = predict_intervals(self.model, img1, img2, meteo_data=meteo_data)
                pred_intervals.append((lower_bound.cpu().numpy(), mean.cpu().numpy(), upper_bound.cpu().numpy()))

                outputs = self.model(img1, img2, meteo_data=meteo_data)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                processed_batches += 1

        # If no intervals were collected, avoid concatenation error
        if len(pred_intervals) == 0:
            print("No valid predictions were made (all batches skipped?).")
            avg_loss = 0.0
            test_time = time.time() - start_test_time
            if self.test_loss_file:
                with open(self.test_loss_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Error', 'Processed Val Batches', 'Skipped Val Batches', 'Testing Time (s)'])
                    writer.writerow([avg_loss, processed_batches, skipped_batches, test_time])
            return avg_loss, processed_batches, skipped_batches

        # Post-processing
        lower_bounds = np.concatenate([pi[0] for pi in pred_intervals], axis=0)
        mean = np.concatenate([pi[1] for pi in pred_intervals], axis=0)
        upper_bounds = np.concatenate([pi[2] for pi in pred_intervals], axis=0)
        true_labels = np.concatenate(true_labels, axis=0)  # Remains 1D

        test_time = time.time() - start_test_time
        avg_loss = running_loss / max(1, processed_batches)

        # Save predictions and labels for analysis
        np.save(self.test_pred_lower_bounds_file, lower_bounds)
        np.save(self.test_pred_mean_file, mean)
        np.save(self.test_pred_upper_bounds_file, upper_bounds)
        np.save(self.test_true_labels_file, true_labels)  # Original labels saved here

        # Log results to test loss file
        if self.test_loss_file:
            with open(self.test_loss_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Error', 'Processed Val Batches', 'Skipped Val Batches', 'Testing Time (s)'])
                writer.writerow([avg_loss, processed_batches, skipped_batches, test_time])

        return avg_loss, processed_batches, skipped_batches