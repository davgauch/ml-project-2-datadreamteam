import torch
import torch.nn as nn
import torch.nn.functional as F


class DualCNN_LSTM(nn.Module):
    def __init__(self,input_shape=(32,6,224,224),out_channels=1, num_meteo_features=5): #input_shape =(32,6,224,224), -> six because 2 images
        """
        Args:
            input_shape: Shape of the input batch used to infer CNN output size. 
                         Default=(32,6,224,224) means: 
                         batch_size=32, channels=6 (because 2 RGB images = 3*2=6),
                         height=224, width=224.
            out_channels: Number of output channels, e.g. for quantile regression, 
                          this might be number of quantiles.
            num_meteo_features: Number of meteorological features to be incorporated.
        """
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

        # MLP for Meteo Data
        # Process the meteo features before concatenation
        self.meteo_fc1 = nn.Linear(num_meteo_features, 32)
        self.meteo_fc2 = nn.Linear(32, 32)
        self.meteo_dropout = nn.Dropout(0.1)

        # LSTM layers
        # Input to LSTM is flattened_size from CNN + 32 from meteo
        self.lstm1 = nn.LSTM(self.flatten_size + 32, 128, batch_first=True)
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

    def forward(self, img1, img2, meteo_data=None, return_features=False):
        """
        Forward pass for two separate images with optional meteo data.
        Args:
            img1 (torch.Tensor): (batch_size, channels, H, W)
            img2 (torch.Tensor): (batch_size, channels, H, W)
            meteo_data (torch.Tensor): (batch_size, num_meteo_features) or None
            return_features (bool): If True, return CNN features only.

        Returns:
            torch.Tensor: Output of shape (batch_size, out_channels).
        """
        # Ensure the images have the same spatial dimensions
        if img1.size() != img2.size():
            raise ValueError("img1 and img2 must have the same shape")
        
        # Concatenate channels of both images: shape (B, C*2, H, W)
        merged_images = torch.cat((img1, img2), dim=1)
        
        # CNN forward
        cnn_out = self.cnn_forward(merged_images)  # (B, features, h', w')
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # Flatten: (B, flattened_size)

        if return_features:
            return cnn_out

        # Process meteo features if provided
        if meteo_data is not None and meteo_data.nelement() > 0:
            # Pass through small MLP
            m = F.relu(self.meteo_fc1(meteo_data))
            m = self.meteo_dropout(m)
            m = F.relu(self.meteo_fc2(m))
            # Concatenate CNN and Meteo features
            combined_input = torch.cat((cnn_out, m), dim=1)
        else:
            combined_input = cnn_out

        # Add a sequence dimension for LSTM
        lstm_in = combined_input.unsqueeze(1)  # (B, 1, feature_dim)

        # LSTM forward
        lstm_out, _ = self.lstm1(lstm_in)
        lstm_out, _ = self.lstm2(lstm_out)

        # Fully-connected layers
        x = F.relu(self.fc1(lstm_out[:, -1, :]))
        x = self.fc2(x)

        return x