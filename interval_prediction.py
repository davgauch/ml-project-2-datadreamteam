import torch

import torch

def predict_intervals(model, inputs):
    """
    Predicts intervals using the trained model. Supports Monte Carlo Dropout.
    Args:
        model (torch.nn.Module): Trained model (Monte Carlo or standard).
        inputs (torch.Tensor): Input data.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds of the prediction interval.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        
        # Handle tuple outputs (Monte Carlo Dropout)
        if isinstance(outputs, tuple):
            mean_prediction, uncertainty = outputs  # Extract mean and uncertainty
            lower_bound = mean_prediction - uncertainty.sqrt()  # Approximate lower bound
            upper_bound = mean_prediction + uncertainty.sqrt()  # Approximate upper bound
        else:
            raise ValueError("Monte Carlo Dropout model must return a tuple (mean, uncertainty).")

    return lower_bound, upper_bound


def display_prediction_intervals(prediction_intervals, num_samples=5):
    """
    Display a summary of the prediction intervals for a subset of samples.

    Args:
        prediction_intervals (list of tuples): The prediction intervals for the batch.
        num_samples (int): The number of samples to display.
    """
    for i, (lower, upper) in enumerate(prediction_intervals[:num_samples]):
        # Calculate the width of the interval
        interval_width = (upper - lower).mean().item()
        print(f"Sample {i+1} - Lower: {lower.mean().item():.3f}, Upper: {upper.mean().item():.3f}, Width: {interval_width:.3f}")
