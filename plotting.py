import matplotlib.pyplot as plt
import numpy as np

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
    Plots the true labels, predicted labels, 95% confidence intervals, and individual MC predictions.
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
    pred_probs_mc = denormalize(pred_probs_mc)

    # Flatten for 1D plotting (if outputs are multi-dimensional, adapt accordingly)
    true_labels = true_labels.flatten()
    pred_mean = pred_mean.flatten()
    lower_bounds = lower_bounds.flatten()
    upper_bounds = upper_bounds.flatten()
    pred_probs_mc = pred_probs_mc.reshape(pred_probs_mc.shape[0], -1)  # Flatten MC predictions

    # Slice the data if num_labels is specified
    if num_labels:
        end_index = start_index + num_labels
        true_labels = true_labels[start_index:end_index]
        pred_mean = pred_mean[start_index:end_index]
        lower_bounds = lower_bounds[start_index:end_index]
        upper_bounds = upper_bounds[start_index:end_index]
        pred_probs_mc = pred_probs_mc[:, start_index:end_index]

    # Plot
    plt.figure(figsize=(12, 6))

    # Plot individual MC samples
    for i in range(pred_probs_mc.shape[0]):  # Iterate over MC samples
        plt.plot(
            range(len(true_labels)),
            pred_probs_mc[i],
            color='red',
            alpha=0.10,
            linestyle='-',
            linewidth=0.5,
            label='_nolegend_' if i > 0 else 'MC Samples'  # Add label only once
        )

    # Plot true labels, mean prediction, and confidence intervals
    plt.plot(range(len(true_labels)), true_labels, label='True Labels', color='blue', linewidth=2)
    plt.plot(range(len(pred_mean)), pred_mean, label='Predicted Labels (Mean)', color='green', linestyle='--', linewidth=2)
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
    plt.title('Predictions with Monte Carlo Samples and 95% Confidence Interval')
    plt.legend()
    plt.grid(True)

    # Save or show plot
    if plot_save_path:
        plt.savefig(plot_save_path, bbox_inches='tight')
        print(f"Plot saved to {plot_save_path}")
    else:
        plt.show()


# Call the function
plot_predictions_bnn(
    preds_file="./bayesianV8Test5/preds.npy",
    true_labels_file="./bayesianV8Test5/true_labels.npy",
    lower_bounds_file="./bayesianV8Test5/pred_lower_bounds.npy",
    upper_bounds_file="./bayesianV8Test5/pred_upper_bounds.npy",
    plot_save_path="bayesianV8Test5.png",
    num_labels=300
)
