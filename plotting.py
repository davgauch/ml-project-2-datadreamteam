import matplotlib.pyplot as plt

def plot_predictions_vs_ground_truth(model, val_loader, device):
    """
    Function to plot predictions vs ground truth for a batch of data from the validation loader.

    Args:
        model (torch.nn.Module): The trained model.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run the model on (cpu or cuda).
    """
    # Iterate through the validation loader (we only plot for one batch)
    for X_batch, y_batch in val_loader:
        # Move the data to the appropriate device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Get model predictions
        outputs = model(X_batch).detach().cpu().numpy()
        labels = y_batch.cpu().numpy()

        # Debugging: Print shapes and check values
        print(f"Outputs: {outputs[:5]}")  # Print first 5 predictions for debugging
        print(f"Labels: {labels[:5]}")    # Print first 5 ground truth values

        # Plot the predictions vs ground truth
        plt.figure(figsize=(10, 6))
        plt.plot(outputs, label='Predictions', color='blue')
        plt.plot(labels, label='Ground Truth', color='red')
        plt.legend()
        plt.title('Predictions vs Ground Truth')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.show()

        break  # Only plot for the first batch
