import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self,input_shape=(32,6,224,224),out_channels=1): #input_shape =(32,3,224,224),
        super().__init__()
        self.batch_size, self.channels = input_shape[0],input_shape[-3]

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # Batch norm layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)

        # Dummy forward pass to get output shape
        with torch.no_grad():
            dummy_input = torch.zeros(*input_shape)  # assuming input_shape is like (batch_size, channels, height, width)
            if len(dummy_input.size()) == 5:
                dummy_input = dummy_input.view(-1, *dummy_input.size()[-3:])
            cnn_out = self.cnn_forward(dummy_input)  # a method that replicates the CNN part of the full forward
            self.flatten_size = cnn_out.view(self.batch_size, -1).size(1)

        # LSTM layers
        self.lstm1 = nn.LSTM(self.flatten_size, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
    
        # Dense layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def cnn_forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = self.dropout(x)
        return x

def forward(self, x1, x2, return_features=False):
    """
    Forward pass for two images x1 and x2.
    x1, x2: Shape (batch_size, n_img, channels, height, width) or (batch_size, channels, height, width)
    """
    merged_images = torch.cat((img1, img2), dim=1)  # Shape: (batch_size, channels*2, height, width)
        
    # Pass the concatenated images through the CNN
    cnn_out = self.cnn_forward(merged_images)  # Shape: (batch_size, cnn_features, reduced_height, reduced_width)

    # Flatten the CNN output
    cnn_out = cnn_out.view(cnn_out.size(0), -1)  # Shape: (batch_size, flattened_size)

    if return_features:
        return combined_features

    # Reshape for LSTM input
    combined_features = combined_features.view(current_batch_size, -1, combined_features.size(-1))

    # LSTM layers
    x, _ = self.lstm1(combined_features)  # output of all timesteps
    x, _ = self.lstm2(x)

    x = F.leaky_relu(self.fc1(x[:, -1, :]))
    x = self.fc2(x)

    return x  # (batch_size, 1)
