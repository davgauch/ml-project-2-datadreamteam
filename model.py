import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self,input_shape=(32,3,224,224),out_channels=1): #input_shape =(32,3,224,224),
        super().__init__()
        self.batch_size, self.channels = input_shape[0],input_shape[-3]
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)

        # keep affine=False,track_running_stats=False -- to avoid error in training
        self.bn1 = nn.BatchNorm2d(16, affine=False, track_running_stats=False)#momentum=0.01)#,eps=1e-3)
        self.bn2 = nn.BatchNorm2d(32, affine=False, track_running_stats=False)#, momentum=0.01)#, momentum=0.01,eps=1e-3)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.1)
        # Dummy forward pass to get output shape
        with torch.no_grad():
            dummy_input = torch.zeros(*input_shape)  # assuming input_shape is like (batch_size, channels, height, width)
            if len(dummy_input.size()) == 5:
                dummy_input = dummy_input.view(-1, *dummy_input.size()[-3:])
            cnn_out = self.cnn_forward(dummy_input)  # a method that replicates the CNN part of the full forward
            self.flatten_size = cnn_out.view(self.batch_size, -1).size(1)

        # LSTM layers
        self.lstm1 = nn.LSTM(self.flatten_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
    
        # Dense layers
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, out_channels)

    def cnn_forward(self, x):
        # x shape: (batch_size, channels, height, width) 
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout2(x)
        return x

def forward(self, x1, x2, return_features=False):
    """
    Forward pass for two images x1 and x2.
    x1, x2: Shape (batch_size, n_img, channels, height, width) or (batch_size, channels, height, width)
    """
    # Handle batch sizes for two images
    if x1.ndim == 5:  # Input is a sequence of images
        current_batch_size, n_img, channels, height, width = x1.size()
        x1 = x1.view(-1, channels, height, width)  # Flatten sequence dimension
        x2 = x2.view(-1, channels, height, width)  # Flatten sequence dimension

    # Pass both images through the CNN
    features_x1 = self.cnn_forward(x1)
    features_x2 = self.cnn_forward(x2)

    # Combine features (concatenation)
    combined_features = torch.cat([features_x1, features_x2], dim=1)  # Concatenate along the feature dimension

    if return_features:
        return combined_features

    # Reshape for LSTM input
    combined_features = combined_features.view(current_batch_size, -1, combined_features.size(-1))

    # LSTM layers
    x, _ = self.lstm1(combined_features)  # output of all timesteps
    x, _ = self.lstm2(x)
    x = F.relu(self.fc1(x[:, -1, :]))  # only last timestep
    x = F.relu(self.fc2(x))

    return x  # (batch_size, 1)
