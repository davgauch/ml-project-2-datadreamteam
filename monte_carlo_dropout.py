import torch
import torch.nn as nn

class MonteCarloDropoutModel(nn.Module):
    def __init__(self, base_model, num_samples=50):
        """
        Extends the base model to enable Monte Carlo Dropout.
        Args:
            base_model (nn.Module): The CNN_LSTM model architecture.
            num_samples (int): Number of stochastic forward passes for MC Dropout.
        """
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples

        # Ensure that dropout layers are present in the base model
        if not any(isinstance(layer, nn.Dropout) for layer in self.base_model.modules()):
            raise ValueError("Base model must contain at least one Dropout layer for Monte Carlo Dropout.")

    def enable_dropout_inference(self):
        """
        Enables dropout during inference by setting dropout layers to training mode.
        """
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active during inference

    def forward(self, x):
        """
        Perform Monte Carlo Dropout to get the mean and 95% confidence interval.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            tuple: (mean_prediction, lower_bound, upper_bound) where:
                - mean_prediction (torch.Tensor): Mean prediction across stochastic forward passes.
                - lower_bound (torch.Tensor): Lower bound of 95% confidence interval.
                - upper_bound (torch.Tensor): Upper bound of 95% confidence interval.
        """
        self.enable_dropout_inference()
        predictions = torch.stack([self.base_model(x) for _ in range(self.num_samples)], dim=0)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)  # Standard deviation for uncertainty
        
        # Calculate 95% confidence intervals
        lower_bound = mean_prediction - 1.96 * uncertainty
        upper_bound = mean_prediction + 1.96 * uncertainty
        
        return mean_prediction, lower_bound, upper_bound
