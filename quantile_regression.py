import torch
import torch.nn as nn

class QuantileRegressionModel(nn.Module):
    def __init__(self, base_model, num_quantiles=2):
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
        else:
            raise AttributeError("Base model must have an 'fc2' layer that is a fully connected layer.")

    def forward(self, x):
        """
        Forward pass through the modified base model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Predicted quantiles.
        """
        outputs = self.base_model(x)
        return outputs
