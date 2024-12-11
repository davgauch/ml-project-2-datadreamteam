import csv
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import numpy as np
from scipy.stats import norm

from early_stopper import EarlyStopper

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, gpu_id, batch_size=32, learning_rate=1e-3, epochs=10, save_every=1, working_dir="/output", num_mc=3, num_monte_carlo=20, model_path=None):
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
            num_mc (int, optional): The number of Monte Carlo runs during training. Default is 15.
            num_monte_carlo (int, optional): The number of Monte Carlo samples to be drawn for inference. Default is 20.
        """
        
        self.gpu_id = gpu_id
        
        self.model = model

        self.start_epoch = 0

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
                loaded_elements.append("EARLY_STOPPER_STATE")
            
            print(f"Model loaded from {model_path}.")
            print(f"Loaded elements: {', '.join(loaded_elements)}")
        else:
            print("No model path provided, starting training from scratch.")

        if torch.cuda.is_available():
            self.model = self.model.to(gpu_id)
            self.model = DDP(model, device_ids=[gpu_id])
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)

            pin_memory = True
        else:
            train_sampler = None
            test_sampler = None
            val_sampler = None

            pin_memory = False

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=val_sampler)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=test_sampler)

        self.epochs = epochs
        self.save_every = save_every

        self.working_dir = working_dir
        self.train_loss_file = f"{working_dir}/train_losses.csv"
        self.test_loss_file = f"{working_dir}/test_loss.csv"
        self.model_snapshot_file = f"{working_dir}/weights.pt"
        self.test_true_labels_file = f"{working_dir}/true_labels.npy"
        self.test_preds_file = f"{working_dir}/preds.npy"
        self.test_pred_lower_bounds_file = f"{working_dir}/pred_lower_bounds.npy"
        self.test_pred_upper_bounds_file = f"{working_dir}/pred_upper_bounds.npy"

        self.num_mc = num_mc
        self.num_monte_carlo = num_monte_carlo

        # Ensure the directories for saving models and loss files exist
        if self.working_dir:
            os.makedirs(self.working_dir, exist_ok=True)

        # Create or overwrite the loss files with headers
        if self.train_loss_file:
            with open(self.train_loss_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Processed Train Batches', 'Skipped Train Batches', 'Processed Val Batches', 'Skipped Val Batches', 'Train Time (s)', 'Eval Time (s)'])

    def train(self):
        """
        Main function to train the model for the specified number of epochs.
        """
        for epoch in range(self.start_epoch, self.epochs):
            # Start timing for training
            start_train_time = time.time()

            # Train for one epoch
            train_loss, processed_train_batches, skipped_train_batches = self.train_one_epoch(epoch)

            # Calculate time taken for training
            train_time = time.time() - start_train_time

            print(f"Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.4f}, Processed Train Batches: {processed_train_batches}, Skipped Train Batches: {skipped_train_batches}, Train Time: {train_time:.2f}s")
            
            # Start timing for evaluation
            start_eval_time = time.time()

            # Evaluate on the validation set
            val_loss, processed_val_batches, skipped_val_batches = self.evaluate()

            # Calculate time taken for evaluation
            eval_time = time.time() - start_eval_time

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch}/{self.epochs}, Validation Loss: {val_loss:.4f}, Processed Validation Batches: {processed_val_batches}, Skipped Validation Batches: {skipped_val_batches}, Eval Time: {eval_time:.2f}s")
            
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
                        "min_validation_loss": self.early_stopper.min_validation_loss,
                    },
                }
                torch.save(snapshot, self.model_snapshot_file)
                print(f"Epoch {epoch} | Training snapshot saved at {self.model_snapshot_file}")

            print(f"Current early_stopper count: {self.early_stopper.counter}")

            if self.early_stopper.early_stop(val_loss):             
                print("Early stopped training.")
                break

        print("Training complete. Losses and batch info saved to:", self.train_loss_file)

    def train_one_epoch(self, epoch):
        """
        Performs one epoch of training on the dataset.
        """
        b_sz = len(next(iter(self.train_loader))[1])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        if torch.cuda.is_available():
            self.train_loader.sampler.set_epoch(epoch)

        self.model.train()

        running_loss = 0.0
        processed_batches = 0
        skipped_batches = 0

        for (img1, img2), labels in self.train_loader:
            # Skip invalid batches (NaN checks, etc.)
            if torch.isnan(img1).any() or torch.isnan(img2).any() or torch.isnan(labels).any():
                skipped_batches += 1
                print("Warning: Skipping batch due to NaN values.")
                continue

            # Reshape labels to (batch_size, 1)
            labels = labels.view(-1, 1)

            # Move inputs and labels to the device (GPU/CPU)
            device = self.gpu_id if torch.cuda.is_available() else "cpu"
            img1, img2, labels = img1.to(device).float(), img2.to(device).float(), labels.to(device).float()
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            output_ = []
            kl_ = []

            for mc_run in range(self.num_mc):
                output = self.model(img1, img2)
                kl = get_kl_loss(self.model)
                output_.append(output)
                kl_.append(kl)

            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)

            # Skip if outputs contain NaN
            if torch.isnan(output).any():
                skipped_batches += 1
                print("Warning: Skipping batch due to NaN in outputs.")
                continue

            # Compute the loss
            criterion_loss = self.criterion(output, labels)
            kl_div = kl / img1.size(0)  # Normalize KL divergence by batch size
            loss = criterion_loss + kl_div

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Collect statistics
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
            for (img1, img2), labels in self.val_loader:
                # Skip invalid batches (NaN checks, etc.)
                if torch.isnan(img1).any() or torch.isnan(img2).any() or torch.isnan(labels).any():
                    skipped_batches += 1
                    print("Warning: Skipping batch due to NaN values.")
                    continue

                # Reshape labels to (batch_size, 1)
                labels = labels.view(-1, 1)

                # Move inputs and labels to the device (GPU/CPU)
                device = self.gpu_id if torch.cuda.is_available() else "cpu"
                img1, img2, labels = img1.to(device).float(), img2.to(device).float(), labels.to(device).float()

                # Forward pass
                output_ = []
                kl_ = []

                for mc_run in range(self.num_mc):
                    output = self.model(img1, img2)
                    kl = get_kl_loss(self.model)
                    output_.append(output)
                    kl_.append(kl)

                output = torch.mean(torch.stack(output_), dim=0)
                kl = torch.mean(torch.stack(kl_), dim=0)

                # Loss
                criterion_loss = self.criterion(output, labels)
                kl_div = kl / img1.size(0)  # Normalize KL divergence by batch size
                loss = criterion_loss + kl_div

                # Collect statistics
                running_loss += loss.item()
                processed_batches += 1
            
        avg_loss = running_loss / max(1, processed_batches)

        return avg_loss, processed_batches, skipped_batches


    def test(self):
        """
        Evaluates the model on the test dataset using Monte Carlo (MC) dropout.
        """
        self.model.eval()
        running_loss = 0.0
        processed_batches = 0
        skipped_batches = 0

        start_test_time = time.time()
        pred_probs_mc = []  # Collect MC predictions
        true_labels = []    # Collect true labels

        lower_bounds = []  # Store lower bounds of 95% CI
        upper_bounds = []  # Store upper bounds of 95% CI

        with torch.no_grad():
            i = 0
            for (img1, img2), labels in self.test_loader:
                i += 1
                # Skip invalid batches (NaN checks, etc.)
                if torch.isnan(img1).any() or torch.isnan(img2).any() or torch.isnan(labels).any():
                    skipped_batches += 1
                    print("Warning: Skipping batch due to NaN values.")
                    continue

                # Reshape labels to (batch_size, 1)
                labels = labels.view(-1, 1)

                # Move inputs and labels to the device (GPU/CPU)
                device = self.gpu_id if torch.cuda.is_available() else "cpu"
                img1, img2, labels = img1.to(device).float(), img2.to(device).float(), labels.to(device).float()

                # Monte Carlo Sampling
                batch_preds_mc = []
                for mc_run in range(self.num_monte_carlo):
                    outputs = self.model(img1, img2)
                    batch_preds_mc.append(outputs.cpu().numpy())

                # Collect MC predictions and true labels
                batch_preds_mc = np.stack(batch_preds_mc, axis=0)  # Shape: (num_monte_carlo, batch_size, 1)
                pred_probs_mc.append(batch_preds_mc)
                true_labels.append(labels.cpu().numpy())

                # Calculate loss using mean prediction
                pred_mean = np.mean(batch_preds_mc, axis=0)  # Mean over MC samples
                pred_std = np.std(batch_preds_mc, axis=0)   # Standard deviation over MC samples

                # Confidence Interval
                z_score = norm.ppf(0.975)  # For 95% confidence
                lower_bound = pred_mean - z_score * pred_std
                upper_bound = pred_mean + z_score * pred_std

                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)

                loss = self.criterion(torch.tensor(pred_mean, dtype=torch.float32), labels.cpu())
                running_loss += loss.item()
                processed_batches += 1

        # Post-processing
        pred_probs_mc = np.concatenate(pred_probs_mc, axis=1)  # Combine all batches
        true_labels = np.concatenate(true_labels, axis=0)
        lower_bounds = np.concatenate(lower_bounds, axis=0)
        upper_bounds = np.concatenate(upper_bounds, axis=0)

        test_time = time.time() - start_test_time
        avg_loss = running_loss / max(1, processed_batches)

        # Save predictions, labels, and confidence intervals for analysis
        np.save(self.test_preds_file, pred_probs_mc)
        np.save(self.test_true_labels_file, true_labels)
        np.save(self.test_pred_lower_bounds_file, lower_bounds)
        np.save(self.test_pred_upper_bounds_file, upper_bounds)

        # Log results to test loss file
        if self.test_loss_file:
            with open(self.test_loss_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['MSE', 'Processed Val Batches', 'Skipped Val Batches', 'Testing Time (s)'])
                writer.writerow([avg_loss, processed_batches, skipped_batches, test_time])

        return avg_loss, processed_batches, skipped_batches