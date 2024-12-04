import torch
import torch.nn as nn

def initialize_weights(layer):
    """
    Initialize the weights of the layer.
    This will initialize the weights using a normal distribution 
    and the biases to zero.
    """
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.LSTM):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)

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
        
        # Initialize only the last layer (fc2) for quantile regression
        initialize_weights(self.base_model.fc2)

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
