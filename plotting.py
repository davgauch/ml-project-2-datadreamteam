import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_vs_ground_truth_mc(model, val_loader, device, num_mc_samples=50):
    """
    Function to plot predictions vs ground truth for Monte Carlo Dropout over a batch of validation data.

    Args:
        model (torch.nn.Module): The trained model with Monte Carlo Dropout enabled.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run the model on (cpu or cuda).
        num_mc_samples (int): Number of Monte Carlo samples to estimate uncertainty.
    """
    for X_batch, y_batch in val_loader:
        # Move the data to the appropriate device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Collect predictions for MC samples
        mc_predictions = []
        for _ in range(num_mc_samples):
            outputs = model(X_batch).detach().cpu().numpy()
            mc_predictions.append(outputs)

        # Compute mean and uncertainty
        mc_predictions = np.stack(mc_predictions, axis=0)  # Shape: (num_mc_samples, batch_size, output_dim)
        mean_predictions = mc_predictions.mean(axis=0)
        std_predictions = mc_predictions.std(axis=0)

        # Compute 95% confidence intervals
        lower_bounds = mean_predictions - 1.96 * std_predictions
        upper_bounds = mean_predictions + 1.96 * std_predictions

        # Plot predictions vs ground truth
        labels = y_batch.cpu().numpy()
        plt.figure(figsize=(12, 6))
        plt.plot(labels, label='Ground Truth', color='red', linestyle='dashed')
        plt.plot(mean_predictions, label='Mean Predictions', color='blue')
        plt.fill_between(
            range(len(mean_predictions)),
            lower_bounds.flatten(),
            upper_bounds.flatten(),
            color='blue',
            alpha=0.2,
            label='95% Confidence Interval',
        )
        plt.legend()
        plt.title('Predictions vs Ground Truth with Monte Carlo Uncertainty')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()

        break  # Only plot for the first batch


def plot_predictions_with_uncertainty_mc(
    preds_file, 
    true_labels_file, 
    lower_bounds_file, 
    upper_bounds_file, 
    plot_save_path=None, 
    num_labels=None, 
    start_index=0
):
    """
    Plot predictions, ground truths, and 95% confidence intervals for Monte Carlo Dropout evaluation.

    Args:
        preds_file (str): Path to the file containing predictions from MC Dropout.
        true_labels_file (str): Path to the file containing true labels.
        lower_bounds_file (str): Path to the file containing lower bounds of the 95% CI.
        upper_bounds_file (str): Path to the file containing upper bounds of the 95% CI.
        plot_save_path (str, optional): File path to save the plot. If None, displays the plot.
        num_labels (int, optional): Number of labels to plot. If None, plots all available data.
        start_index (int, optional): The starting index for the subset of labels to plot.
    """
    # Load data
    predictions = np.load(preds_file)  # Shape: (num_samples, output_dim)
    true_labels = np.load(true_labels_file)  # Shape: (num_samples, output_dim)
    lower_bounds = np.load(lower_bounds_file)  # Shape: (num_samples, output_dim)
    upper_bounds = np.load(upper_bounds_file)  # Shape: (num_samples, output_dim)

    # Flatten for 1D plotting if needed
    predictions = predictions.flatten()
    true_labels = true_labels.flatten()
    lower_bounds = lower_bounds.flatten()
    upper_bounds = upper_bounds.flatten()

    # Slice the data for visualization
    if num_labels:
        end_index = start_index + num_labels
        predictions = predictions[start_index:end_index]
        true_labels = true_labels[start_index:end_index]
        lower_bounds = lower_bounds[start_index:end_index]
        upper_bounds = upper_bounds[start_index:end_index]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(true_labels, label='True Labels', color='blue', linewidth=2)
    plt.plot(predictions, label='Predictions', color='green', linestyle='--', linewidth=2)
    plt.fill_between(
        range(len(predictions)),
        lower_bounds,
        upper_bounds,
        color='gray',
        alpha=0.3,
        label='95% Confidence Interval'
    )
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predictions vs Ground Truth with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)

    if plot_save_path:
        plt.savefig(plot_save_path, bbox_inches='tight')
        print(f"Plot saved to {plot_save_path}")
    else:
        plt.show()
