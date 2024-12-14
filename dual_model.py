import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class Attention(nn.Module):
    """Attention mechanism for sequence modeling."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        weights = F.softmax(self.attn(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # Weighted sum (B, hidden_size)
        return context

class DualCNN_LSTM(nn.Module):
    def __init__(self, input_shape=(32, 6, 224, 224), out_channels=1, num_meteo_features=5):
        super().__init__()
        self.num_meteo_features = num_meteo_features

        # Replace custom CNN with a pretrained ResNet
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final classification layer

        # Modify the first convolutional layer to accept 6 channels
        original_conv1 = self.cnn.conv1
        self.cnn.conv1 = nn.Conv2d(6, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
                                   stride=original_conv1.stride, padding=original_conv1.padding, bias=original_conv1.bias)
        with torch.no_grad():
            self.cnn.conv1.weight[:, :3] = original_conv1.weight  # Copy weights for the first 3 channels (RGB)
            self.cnn.conv1.weight[:, 3:] = original_conv1.weight  # Duplicate weights for the additional 3 channels

        # Determine the flatten size dynamically
        dummy_input = torch.zeros(1, input_shape[1], input_shape[2], input_shape[3])
        self.flatten_size = self.cnn(dummy_input).view(-1).size(0)

        # MLP for Meteo Data
        self.meteo_fc1 = nn.Linear(num_meteo_features, 32)
        self.meteo_fc2 = nn.Linear(32, 32)
        self.meteo_dropout = nn.Dropout(0.1)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(self.flatten_size + 32, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)

        # Attention mechanism
        self.attention = Attention(hidden_size=128)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, out_channels)

    def forward(self, img1, img2, meteo_data=None, return_features=False):
        # Ensure the images have the same spatial dimensions
        if img1.size() != img2.size():
            raise ValueError("img1 and img2 must have the same shape")

        # Concatenate channels of both images: shape (B, C*2, H, W)
        merged_images = torch.cat((img1, img2), dim=1)

        # CNN forward
        cnn_out = self.cnn(merged_images)  # (B, flattened_size)

        if return_features:
            return cnn_out

        # Process meteorological data
        if meteo_data is not None and meteo_data.nelement() > 0:
            m = F.relu(self.meteo_fc1(meteo_data))
            m = self.meteo_dropout(m)
            m = F.relu(self.meteo_fc2(m))
            combined_input = torch.cat((cnn_out, m), dim=1)
        else:
            combined_input = cnn_out

        # Add a sequence dimension for LSTM
        lstm_in = combined_input.unsqueeze(1)  # (B, 1, feature_dim)

        # LSTM forward
        lstm_out, _ = self.lstm1(lstm_in)
        lstm_out, _ = self.lstm2(lstm_out)

        # Apply attention
        context = self.attention(lstm_out)  # (B, hidden_size)

        # Fully-connected layers
        x = F.relu(self.fc1(context))
        x = self.fc2(x)

        return x
