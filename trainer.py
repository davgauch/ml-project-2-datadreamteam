import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size=32, learning_rate=1e-3, epochs=10, device="cpu"):
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
        """
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
    
    def train_one_epoch(self):
        """
        Performs one epoch of training on the dataset.
        """
        self.model.train()
        running_loss = 0.0
        skipped_batches = 0

        for inputs, labels in self.train_loader:
            
            # Check batch size
            if inputs.size(0) != 32:
                print(f"Warning: Unexpected batch size {inputs.size()}. Skipping batch.")
                skipped_batches += 1
                continue
            
            # Check channels size
            if inputs.size(1) != 3:  # Example: check if channels = 3
                print(f"Warning: Unexpected channels size {inputs.size()}. Skipping batch.")
                skipped_batches += 1
                continue
            
            # Check images size
            if inputs.size(2) != 250 or inputs.size(3) != 250:
                print(f"Warning: Unexpected image size {inputs.size()}. Skipping batch.")
                skipped_batches += 1
                continue
            
            # Check that batch size of inputs matches labels
            if labels.size(0) != inputs.size(0):  
                print(f"Warning: Mismatch between input batch size and label batch size: {inputs.size(0)} vs {labels.size(0)}")
                skipped_batches += 1
                continue

            # Skip batches with NaN
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                skipped_batches += 1
                print("Warning: Skipping batch due to NaN values.")
                continue

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

            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Collect statistics
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(self.train_loader) - skipped_batches)
        print(f"Skipped {skipped_batches} batches due to NaN.")
        return avg_loss

    def evaluate(self):
        """
        Evaluates the model on the validation dataset.
        """
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        """
        Main function to train the model for the specified number of epochs.
        """
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss = self.train_one_epoch()
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}")
            
            # Evaluate on the validation set
            val_loss = self.evaluate()
            print(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {val_loss:.4f}")
            
            # Saving logic here (e.g., save model after every epoch)
            # torch.save(self.model.state_dict(), f"model_epoch_{epoch+1}.pth")
            
        print("Training complete.")

    def test(self, max_batches=None):
        """
        Test the model on the test set and print results.
        
        Args:
            max_batches (int, optional): Maximum number of batches to process during testing. If None, process all batches.
        """
        self.model.eval()
        test_loss = 0.0
        batch_count = 0  # Counter to track number of batches processed
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                if max_batches is not None and batch_count >= max_batches:
                    break  # Exit the loop if maximum batches have been processed

                # Check batch size
                if inputs.size(0) != 32:
                    print(f"Warning: Unexpected batch size {inputs.size()}. Skipping batch.")
                    continue
                
                # Check channels size
                if inputs.size(1) != 3:
                    print(f"Warning: Unexpected channels size {inputs.size()}. Skipping batch.")
                    continue
                
                # Check images size
                if inputs.size(2) != 250 or inputs.size(3) != 250:
                    print(f"Warning: Unexpected image size {inputs.size()}. Skipping batch.")
                    continue
                
                # Check that batch size of inputs matches labels
                if labels.size(0) != inputs.size(0):  
                    print(f"Warning: Mismatch between input batch size and label batch size: {inputs.size(0)} vs {labels.size(0)}")
                    continue

                # Skip batches with NaN
                if torch.isnan(inputs).any() or torch.isnan(labels).any():
                    skipped_batches += 1
                    print("Warning: Skipping batch due to NaN values.")
                    continue
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                batch_count += 1  

        # Handle the case where no batches were processed to avoid division by zero
        avg_test_loss = test_loss / max(1, batch_count)
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Processed {batch_count} batches during testing.")
