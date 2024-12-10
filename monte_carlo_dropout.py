import torch
import torch.nn as nn

class MonteCarloDropoutModel(nn.Module):
    def __init__(self, base_model, mc_samples=50):
        """
        Extends the base model to enable Monte Carlo Dropout.

        Args:
            base_model (nn.Module): The base model with Dropout layers (e.g., CNN_LSTM).
            mc_samples (int): Number of stochastic forward passes for MC Dropout.
        """
        super(MonteCarloDropoutModel, self).__init__()
        self.base_model = base_model
        self.mc_samples = mc_samples

        # Ensure the base model contains at least one Dropout layer
        if not any(isinstance(layer, nn.Dropout) for layer in self.base_model.modules()):
            raise ValueError("Base model must contain at least one Dropout layer for Monte Carlo Dropout.")

    def enable_dropout_inference(self):
        """
        Enables dropout during inference by setting dropout layers to training mode.
        This ensures stochasticity during forward passes.
        """
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep Dropout active even in eval mode

    def forward(self, x):
        """
        Perform Monte Carlo Dropout to compute predictions with uncertainty estimates.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).

        Returns:
            tuple: (mean_prediction, lower_bound, upper_bound) where:
                - mean_prediction (torch.Tensor): Mean prediction across stochastic forward passes.
                - lower_bound (torch.Tensor): Lower bound of 95% confidence interval.
                - upper_bound (torch.Tensor): Upper bound of 95% confidence interval.
        """
        # Enable Dropout layers during inference
        self.enable_dropout_inference()

        # Perform stochastic forward passes
        predictions = torch.stack([self.base_model(x) for _ in range(self.mc_samples)], dim=0)  # Shape: (mc_samples, batch_size, ...)

        # Compute mean and uncertainty (standard deviation)
        mean_prediction = predictions.mean(dim=0)      # Shape: (batch_size, ...)
        uncertainty = predictions.std(dim=0)           # Shape: (batch_size, ...)

        # Compute 95% confidence intervals
        z_score = 1.96  # For 95% confidence
        lower_bound = mean_prediction - z_score * uncertainty
        upper_bound = mean_prediction + z_score * uncertainty

        return mean_prediction, lower_bound, upper_bound
