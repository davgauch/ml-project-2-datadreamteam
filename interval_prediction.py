import torch

def predict_intervals(model, inputs):
    """
    Predicts intervals using the trained quantile regression model.
    Args:
        model (torch.nn.Module): Trained quantile regression model.
        inputs (torch.Tensor): Input data.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds of the prediction interval.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    lower_bound = torch.clamp(outputs[:, 0], min=0.0)# Lower quantile (GHI cannot be negative)
    upper_bound = outputs[:, 1]  # Upper quantile
    return lower_bound, upper_bound


def display_prediction_intervals(prediction_intervals, num_samples=5):
    """
    Display a summary of the prediction intervals for a subset of samples.

    Args:
        prediction_intervals (list of tuples): The prediction intervals for the batches.
        num_samples (int): The number of samples to display.
    """
    for i, (lower, upper) in enumerate(prediction_intervals[:num_samples]):
        for j in range(min(len(lower), num_samples)):  # Display up to `num_samples` per batch
            interval_width = (upper[j] - lower[j]).item()
            print(f"Sample {j+1} - Lower: {lower[j]:.3f}, Upper: {upper[j]:.3f}, Width: {interval_width:.3f}")
