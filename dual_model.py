import torch
import torch.nn as nn
import torch.nn.functional as F
from initialization import initialize_weights

class DualCNN_LSTM(nn.Module):
    def __init__(self, input_shape=(32, 3, 224, 224), out_channels=1):  # input_shape = (batch_size, channels, height, width)
        super().__init__()
        self.batch_size, self.channels = input_shape[0], input_shape[-3]

        # CNN layers for Webcam 1
        self.cnn1 = self._build_cnn_branch(self.channels)

        # CNN layers for Webcam 2
        self.cnn2 = self._build_cnn_branch(self.channels)

        # Dummy forward pass to determine flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(*input_shape)  # shape = (batch_size, channels, height, width)
            cnn_out_1 = self.cnn1(dummy_input)
            cnn_out_2 = self.cnn2(dummy_input)
            combined_out = torch.cat([cnn_out_1, cnn_out_2], dim=1)
            self.flatten_size = combined_out.view(self.batch_size, -1).size(1)

        # LSTM layers
        self.lstm1 = nn.LSTM(self.flatten_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)

        # Dense layers
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, out_channels)

        self.apply(initialize_weights)

    def _build_cnn_branch(self, channels):
        """Builds a single CNN branch."""
        layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1)
        )
        return layers

    def forward(self, x1, x2, return_features=False):
        """
        x1, x2: Tensors of shape (batch_size, num_images, channels, height, width)
        """
        # Process Webcam 1 input
        if x1.ndim == 5:
            current_batch_size, n_img, channels, height, width = x1.size()
            x1 = x1.view(-1, channels, height, width)
            x1 = self.cnn1(x1)
            x1 = x1.view(current_batch_size, n_img, -1)
        else:
            current_batch_size = x1.size(0)
            x1 = self.cnn1(x1)

        # Process Webcam 2 input
        if x2.ndim == 5:
            current_batch_size, n_img, channels, height, width = x2.size()
            x2 = x2.view(-1, channels, height, width)
            x2 = self.cnn2(x2)
            x2 = x2.view(current_batch_size, n_img, -1)
        else:
            current_batch_size = x2.size(0)
            x2 = self.cnn2(x2)

        # Combine CNN outputs
        x = torch.cat([x1, x2], dim=-1)

        if return_features:
            return x

        # LSTM processing
        x = x.reshape(current_batch_size, 1, -1)
        x, _ = self.lstm1(x)  # Output of all timesteps
        x, _ = self.lstm2(x)

        # Fully connected layers
        x = F.relu(self.fc1(x[:, -1, :]))  # Only the last timestep
        x = self.fc2(x)

        return x  # (batch_size, 1)
