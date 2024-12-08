import matplotlib.pyplot as plt
import numpy as np

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

def denormalize(normalized_value, min_val=1.7, max_val=1027.0):
    """
    Denormalize the value using the min-max normalization formula.
    
    Args:
        normalized_value (float or np.array): The normalized value(s) to denormalize.
        min_val (float): The minimum value used for normalization (default 1.7).
        max_val (float): The maximum value used for normalization (default 1027.0).
    
    Returns:
        float or np.array: The denormalized value(s).
    """
    return normalized_value * (max_val - min_val) + min_val


def plot_predictions_bnn(
    preds_file, 
    true_labels_file, 
    lower_bounds_file, 
    upper_bounds_file, 
    plot_save_path=None, 
    num_labels=None,  # Specify the number of labels to plot
    start_index=0      # Optionally specify a starting index
):
    """
    Plots the true labels, predicted labels, and 95% confidence intervals for a subset of data.

    Args:
        preds_file (str): Path to the file containing predicted probabilities from MC sampling.
        true_labels_file (str): Path to the file containing true labels.
        lower_bounds_file (str): Path to the file containing lower bounds of the 95% CI.
        upper_bounds_file (str): Path to the file containing upper bounds of the 95% CI.
        plot_save_path (str, optional): Path to save the generated plot. If None, the plot is displayed.
        num_labels (int, optional): Number of labels to plot. If None, plots all available data.
        start_index (int, optional): The starting index for the subset of labels to plot.
    """
    # Load data
    pred_probs_mc = np.load(preds_file)  # Shape: (num_monte_carlo, total_samples, output_dim)
    true_labels = np.load(true_labels_file)  # Shape: (total_samples, output_dim)
    lower_bounds = np.load(lower_bounds_file)  # Shape: (total_samples, output_dim)
    upper_bounds = np.load(upper_bounds_file)  # Shape: (total_samples, output_dim)

    # Compute mean prediction across MC samples
    pred_mean = np.mean(pred_probs_mc, axis=0)  # Shape: (total_samples, output_dim)

    # Denormalize the values
    true_labels = denormalize(true_labels)
    pred_mean = denormalize(pred_mean)
    lower_bounds = denormalize(lower_bounds)
    upper_bounds = denormalize(upper_bounds)

    # Flatten for 1D plotting (if outputs are multi-dimensional, adapt accordingly)
    true_labels = true_labels.flatten()
    pred_mean = pred_mean.flatten()
    lower_bounds = lower_bounds.flatten()
    upper_bounds = upper_bounds.flatten()

    # Slice the data if num_labels is specified
    if num_labels:
        end_index = start_index + num_labels
        true_labels = true_labels[start_index:end_index]
        pred_mean = pred_mean[start_index:end_index]
        lower_bounds = lower_bounds[start_index:end_index]
        upper_bounds = upper_bounds[start_index:end_index]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(true_labels, label='True Labels', color='blue', linewidth=2)
    plt.plot(pred_mean, label='Predicted Labels (Mean)', color='green', linestyle='--', linewidth=2)
    plt.fill_between(
        range(len(true_labels)),
        lower_bounds,
        upper_bounds,
        color='gray',
        alpha=0.3,
        label='95% Confidence Interval'
    )

    # Add labels, title, and legend
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Value')
    plt.title('Predictions with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)

    # Save or show plot
    if plot_save_path:
        plt.savefig(plot_save_path, bbox_inches='tight')
        print(f"Plot saved to {plot_save_path}")
    else:
        plt.show()


plot_predictions_bnn(
    preds_file="./bayesianV5Test/preds.npy",
    true_labels_file="./bayesianV5Test/true_labels.npy",
    lower_bounds_file="./bayesianV5Test/pred_lower_bounds.npy",
    upper_bounds_file="./bayesianV5Test/pred_upper_bounds.npy",
    plot_save_path="bayesianV5Test.png",
    num_labels=300
)