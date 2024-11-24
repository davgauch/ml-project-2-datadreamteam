import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels=3, lstm_hidden_size=128, num_lstm_layers=2, output_size=1):
        super().__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3)  # First convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  # Second convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)  # Third convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        
        # We need to compute the final feature size after convolutions and pooling
        # Input size is (3, 250, 250)
        # After 3 layers of conv + pooling, the size becomes:
        # (250 // 2 // 2 // 2) = 31 (height and width) after 3 maxpool operations
        self.fc1_input_dim = 128 * 29 * 29  # 128 channels, 29x29 spatial dimensions
        
        # Fully connected layer for regression (output layer)
        self.fc1 = nn.Linear(self.fc1_input_dim, output_size)  # Output a single regression value

    def forward(self, x):
        # Apply convolutional layers with activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output to feed it into the fully connected layer
        x = x.view(32, -1)  # Flatten to [batch_size, 128 * 29 * 29]

        # Pass through the fully connected layer
        x = self.fc1(x)  # Output: [batch_size, 1]
        print(x.mean())
        
        return x
