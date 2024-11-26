import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class WebcamDataset(Dataset):
    """
    Custom Dataset class for loading webcam images and corresponding GHI (Global Horizontal Irradiance) values.
    This implementation uses lazy loading to avoid loading all the data into memory.
    """

    def __init__(self, images_path, ghi_values_path, transform=None, subset=None, normalize_labels=True):
        """
        Initializes the WebcamDataset object by loading the image data and GHI values from the provided file paths.

        Args:
            images_path (str): Path to the `.npy` file containing the webcam images.
            ghi_values_path (str): Path to the `.npy` file containing the corresponding GHI values.
            transform (callable, optional): Optional transformation to apply on the image data (e.g., normalization, augmentation).
            subset (int, optional): The number of samples to randomly sample from the dataset.
        """
        # Load the image data and GHI values from .npy files
        self.images = np.load(images_path, mmap_mode='r')
        self.ghi_values = np.load(ghi_values_path)

        # Remove rows where any NaN values are present in the images or ghi_values
        # valid_indices = ~np.isnan(self.ghi_values) & ~np.isnan(self.images).any(axis=(1, 2, 3))
        # self.images = self.images[valid_indices]
        # self.ghi_values = self.ghi_values[valid_indices]

        # Perform random subsampling if subset is specified
        if subset:
            indices = np.random.choice(len(self.images), size=subset, replace=False)
            self.images = self.images[:subset]
            self.ghi_values = self.ghi_values[:subset]

        # Compute normalization parameters for images
        # self.image_mean = self.images.mean(axis=(0, 1, 2)) / 255.0  # Normalize by 255
        # self.image_std = self.images.std(axis=(0, 1, 2)) / 255.0

        # # Normalize labels if required
        # self.normalize_labels = normalize_labels
        # if normalize_labels:
        #     self.ghi_min = self.ghi_values.min()
        #     self.ghi_max = self.ghi_values.max()
        #     self.ghi_values = (self.ghi_values - self.ghi_min) / (self.ghi_max - self.ghi_min)

        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.ghi_values.shape[0]

    def __getitem__(self, idx):
        """
        Fetches the image and corresponding GHI value at the specified index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The webcam image at the specified index, converted to a tensor.
                - ghi_value (float): The GHI value corresponding to the image at the specified index.
        """
        # Get the image and the corresponding GHI value
        image = self.images[idx]  # Shape: (height, width, channels)
        ghi_value = self.ghi_values[idx]  # Shape: () - a scalar

        # Normalize the image
        # image = (image - self.image_mean) / self.image_std
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0)

        ghi_value = torch.tensor(ghi_value, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, ghi_value  # Return the image and its corresponding GHI value
