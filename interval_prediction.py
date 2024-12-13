import torch

def predict_intervals(model, img1, img2, meteo_data=None):
    """
    Predicts intervals using the trained quantile regression model.
    
    Args:
        model (torch.nn.Module): Trained quantile regression model.
        img1 (torch.Tensor): First image input.
        img2 (torch.Tensor): Second image input.
        meteo_data (torch.Tensor, optional): Meteorological data. Shape: (batch_size, num_meteo_features).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            lower_bound (torch.Tensor): Lower quantile predictions.
            mean (torch.Tensor): Median (or mean) predictions.
            upper_bound (torch.Tensor): Upper quantile predictions.
    """
    model.eval()
    with torch.no_grad():
        if meteo_data is not None:
            outputs = model(img1, img2, meteo_data=meteo_data)
        else:
            outputs = model(img1, img2)

    lower_bound = torch.clamp(outputs[:, 0], min=0.0)  # Lower quantile (GHI cannot be negative)
    mean = outputs[:, 1]  # Median quantile
    upper_bound = outputs[:, 2]  # Upper quantile

    return lower_bound, mean, upper_bound
