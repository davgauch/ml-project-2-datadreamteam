import torch
import torch.nn as nn
from initialization import initialize_weights

class QuantileRegressionModel(nn.Module):
    def __init__(self, base_model, num_quantiles=3):
        """
        Extends the base CNN_LSTM model for quantile regression.
        Args:
            base_model (nn.Module): The CNN_LSTM model architecture.
            num_quantiles (int): Number of quantiles to predict.
        """
        super().__init__()
        self.base_model = base_model
        
        # Modify the final fully connected layer to output the required number of quantiles
        if hasattr(base_model, 'fc2') and isinstance(base_model.fc2, nn.Linear):
            in_features = base_model.fc2.in_features
            self.base_model.fc2 = nn.Linear(in_features, num_quantiles)
            initialize_weights(self.base_model.fc2) 
        else:
            raise AttributeError("Base model must have an 'fc2' layer that is a fully connected layer.")

    def forward(self, x1, x2, meteo_data=None):
        """
        Forward pass through the modified base model.
        Args:
            x1, x2 (torch.Tensor): Image tensors.
            meteo_data (torch.Tensor, optional): Meteorological feature tensor.
        Returns:
            torch.Tensor: Predicted quantiles.
        """
        outputs = self.base_model(x1, x2, meteo_data=meteo_data)
        return outputs
