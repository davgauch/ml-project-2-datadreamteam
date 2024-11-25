import csv
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size=32, learning_rate=1e-3, epochs=10, device="cpu", model_save_folder="output/model", train_loss_file="train_losses.csv", test_loss_file="test_losses.csv"):
        """
        Initializes the Trainer class with all the necessary components.
        
        Args:
            model (torch.nn.Module): The model to train.
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            test_dataset (Dataset): The test dataset.
            batch_size (int): The batch size to use during training.
            learning_rate (float): The learning rate for the optimizer.
            epochs (int): The number of epochs to train for.
            loss_file (str): Path to the file where losses will be saved (CSV format).
        """
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

        self.train_loss_file = train_loss_file 
        self.test_loss_file = test_loss_file 
        self.model_save_folder = model_save_folder

        # Ensure the directories for saving models and loss files exist
        if model_save_folder:
            os.makedirs(self.model_save_folder, exist_ok=True)

        if self.train_loss_file:
            os.makedirs(os.path.dirname(self.train_loss_file), exist_ok=True)

        if self.test_loss_file:
            os.makedirs(os.path.dirname(self.test_loss_file), exist_ok=True)

        # Create or overwrite the loss files with headers
        if self.train_loss_file:
            with open(self.train_loss_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Processed Train Batches', 'Skipped Train Batches', 'Processed Val Batches', 'Skipped Val Batches', 'Train Time (s)', 'Eval Time (s)'])

    def train(self):
        """
        Main function to train the model for the specified number of epochs.
        """
        for epoch in range(self.epochs):
            # Start timing for training
            start_train_time = time.time()

            # Train for one epoch
            train_loss, processed_train_batches, skipped_train_batches = self.train_one_epoch()

            # Calculate time taken for training
            train_time = time.time() - start_train_time

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Processed Train Batches: {processed_train_batches}, Skipped Train Batches: {skipped_train_batches}, Train Time: {train_time:.2f}s")
            
            # Start timing for evaluation
            start_eval_time = time.time()

            # Evaluate on the validation set
            val_loss, processed_val_batches, skipped_val_batches = self.evaluate()

            # Calculate time taken for evaluation
            eval_time = time.time() - start_eval_time

            print(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {val_loss:.4f}, Processed Validation Batches: {processed_val_batches}, Skipped Validation Batches: {skipped_val_batches}, Eval Time: {eval_time:.2f}s")
            
            # Store losses and batch info for this epoch
            epoch_results = [epoch + 1, train_loss, val_loss, processed_train_batches, skipped_train_batches, processed_val_batches, skipped_val_batches, train_time, eval_time]
            
            # Immediately write results to the CSV file after each epoch
            if self.train_loss_file:
                with open(self.train_loss_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(epoch_results)

            # Saving the model after each epoch
            if self.model_save_folder:
                torch.save(self.model.state_dict(), f"{self.model_save_folder}model_epoch_{epoch+1}.pth")

        print("Training complete. Losses and batch info saved to:", self.train_loss_file)

    def train_one_epoch(self):
        """
        Performs one epoch of training on the dataset.
        """
        self.model.train()
        running_loss = 0.0
        processed_batches = 0
        skipped_batches = 0

        for inputs, labels in self.train_loader:
            # Skip invalid batches (NaN checks, etc.)
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                skipped_batches += 1
                print("Warning: Skipping batch due to NaN values.")
                continue

            # Reshape labels to (batch_size, 1)
            labels = labels.view(-1, 1)

            # Move inputs and labels to the device (GPU/CPU)
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Skip if outputs contain NaN
            if torch.isnan(outputs).any():
                skipped_batches += 1
                print("Warning: Skipping batch due to NaN in outputs.")
                continue

            # Compute the loss
            loss = self.criterion(outputs, labels)

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

        start_test_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                # Skip invalid batches (NaN checks, etc.)
                if torch.isnan(inputs).any() or torch.isnan(labels).any():
                    skipped_batches += 1
                    print("Warning: Skipping batch due to NaN values.")
                    continue

                # Reshape labels to (batch_size, 1)
                labels = labels.view(-1, 1)

                # Move inputs and labels to the device (GPU/CPU)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Collect statistics
                running_loss += loss.item()
                processed_batches += 1
            
        # Calculate time taken for testing
        test_time = time.time() - start_test_time

        avg_loss = running_loss / max(1, processed_batches)

        # Log results to test loss file
        if self.test_loss_file:
            with open(self.test_loss_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['MSE', 'Processed Val Batches', 'Skipped Val Batches', 'Testing Time (s)'])
                writer.writerow([avg_loss, processed_batches, skipped_batches, test_time])

        return avg_loss, processed_batches, skipped_batches
