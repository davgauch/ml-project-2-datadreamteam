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


def plot_predictions_quantile(
    true_labels_file, 
    lower_bounds_file,
    mean_quantiles_file,
    upper_bounds_file, 
    plot_save_path=None, 
    num_labels=None,  # Specify the number of labels to plot
    start_index=0      # Optionally specify a starting index
):
    """
    Plots the true labels, predicted intervals, and midpoint predictions for quantile regression.

    Args:
        true_labels_file (str): Path to the file containing true labels.
        lower_bounds_file (str): Path to the file containing lower bounds of the prediction interval.
        upper_bounds_file (str): Path to the file containing upper bounds of the prediction interval.
        plot_save_path (str, optional): Path to save the generated plot. If None, the plot is displayed.
        num_labels (int, optional): Number of labels to plot. If None, plots all available data.
        start_index (int, optional): The starting index for the subset of labels to plot.
    """
    # Load data
    true_labels = np.load(true_labels_file)  # Shape: (total_samples,)
    lower_bounds = np.load(lower_bounds_file)  # Shape: (total_samples,)
    mean_quantiles = np.load(mean_quantiles_file)    # Shape: (total_samples,)
    upper_bounds = np.load(upper_bounds_file)  # Shape: (total_samples,)
    
    # Print a few samples to check for duplication
    print("True Labels:", true_labels[:5])
    print("Lower Bounds:", lower_bounds[:5])
    print("Mean:", mean_quantiles[:5])
    print("Upper Bounds:", upper_bounds[:5])


    # Denormalize the values if needed
    true_labels = denormalize(true_labels)
    lower_bounds = denormalize(lower_bounds)
    mean_quantiles = denormalize(mean_quantiles)
    upper_bounds = denormalize(upper_bounds)

    # Slice the data if num_labels is specified
    if num_labels:
        end_index = start_index + num_labels
        true_labels = true_labels[start_index:end_index]
        lower_bounds = lower_bounds[start_index:end_index]
        mean_quantiles = mean_quantiles[start_index:end_index]
        upper_bounds = upper_bounds[start_index:end_index]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(true_labels, label='True Labels', color='blue', linewidth=2)
    plt.plot(mean_quantiles, label='Mean Quantile (Prediction)', color='green', linestyle='--', linewidth=2)
    plt.fill_between(
        range(len(true_labels)),
        lower_bounds,
        upper_bounds,
        color='gray',
        alpha=0.3,
        label='Prediction Interval'
    )

    # Add labels, title, and legend
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Value')
    plt.title('Predictions with Prediction Interval')
    plt.legend()
    plt.grid(True)

    # Save or show plot
    if plot_save_path:
        plt.savefig(plot_save_path, bbox_inches='tight')
        print(f"Plot saved to {plot_save_path}")
    else:
        plt.show()



plot_predictions_quantile(
    true_labels_file="./results/output_quantile_v11/true_labels.npy",
    lower_bounds_file="./results/output_quantile_v11/pred_lower_bounds.npy",
    mean_quantiles_file="./results/output_quantile_v11/pred_mean.npy",
    upper_bounds_file="./results/output_quantile_v11/pred_upper_bounds.npy",
    plot_save_path="quantileTestPlot_v11_100.png",
    num_labels=300
)
