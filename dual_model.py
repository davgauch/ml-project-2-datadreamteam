import torch
import torch.nn as nn
import torch.nn.functional as F


class DualCNN_LSTM(nn.Module):
    def __init__(self,input_shape=(32,6,224,224),out_channels=1): #input_shape =(32,6,224,224), -> six because 2 images
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

    def forward(self, img1, img2, return_features=False):
        """
        Forward pass for two separate images with channel concatenation.
        Args:
            img1: Tensor of shape (batch_size, channels, height, width) for the first image.
            img2: Tensor of shape (batch_size, channels, height, width) for the second image.
            return_features: If True, return CNN features instead of final output.
        Returns:
            Output tensor of shape (batch_size, out_channels).
        """
        # Ensure the images have the same spatial dimensions
        if img1.size() != img2.size():
            raise ValueError("img1 and img2 must have the same shape")
        
        # Concatenate the channels of both images
        merged_images = torch.cat((img1, img2), dim=1)  # Shape: (batch_size, channels*2, height, width)
        
        # Pass the concatenated images through the CNN
        cnn_out = self.cnn_forward(merged_images)  # Shape: (batch_size, cnn_features, reduced_height, reduced_width)
        
        # Flatten the CNN output
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # Shape: (batch_size, flattened_size)
        
        if return_features:
            return cnn_out  # Return the features if requested

        # Add a sequence dimension for the LSTM
        lstm_in = cnn_out.unsqueeze(1)  # Shape: (batch_size, seq_len=1, feature_dim)

        # Pass through LSTM layers
        lstm_out, _ = self.lstm1(lstm_in)  # Shape: (batch_size, seq_len, 128)
        lstm_out, _ = self.lstm2(lstm_out)  # Shape: (batch_size, seq_len, 64)

        # Use the last time step for classification
        x = F.relu(self.fc1(lstm_out[:, -1, :]))  # Shape: (batch_size, 64)
        x = F.relu(self.fc2(x))  # Shape: (batch_size, out_channels)

        return x