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
    mean = outputs[:, 1]
    upper_bound = outputs[:, 2]  # Upper quantile
    return lower_bound, mean, upper_bound

