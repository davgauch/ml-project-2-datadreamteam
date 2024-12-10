import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class WebcamDataset(Dataset):
    """
    Custom Dataset class for loading webcam images and corresponding GHI (Global Horizontal Irradiance) values.
    This implementation uses lazy loading to avoid loading all the data into memory.
    """

    def __init__(self, images_path_bc, images_path_m, ghi_values_path, transform=None, subset=None, normalize_labels=True):
        """
        Initializes the WebcamDataset object by loading the image data and GHI values from the provided file paths.

        Args:
            images_path (str): Path to the `.npy` file containing the webcam images.
            ghi_values_path (str): Path to the `.npy` file containing the corresponding GHI values.
            transform (callable, optional): Optional transformation to apply on the image data (e.g., normalization, augmentation).
            subset (int, optional): The number of samples to randomly sample from the dataset.
        """
        # Load the image data and GHI values from .npy files
        self.images_bc = np.load(images_path_bc, mmap_mode='r')
        self.images_m = np.load(images_path_m, mmap_mode='r')
        self.ghi_values = np.load(ghi_values_path)

        # Remove rows where any NaN values are present in the images or ghi_values
        # valid_indices = ~np.isnan(self.ghi_values) & ~np.isnan(self.images).any(axis=(1, 2, 3))
        # self.images = self.images[valid_indices]
        # self.ghi_values = self.ghi_values[valid_indices]

        # Perform random subsampling if subset is specified
        if subset:
            indices = np.random.choice(len(self.images_bc), size=subset, replace=False)
            self.images_bc = self.images_bc[:subset]
            self.images_m = self.images_m[:subset]
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
        image_bc = self.images_bc[idx]  # Shape: (height, width, channels)
        image_m = self.images_m[idx]  # Shape: (height, width, channels)
        ghi_value = self.ghi_values[idx]  # Shape: () - a scalar

        # Normalize the image
        # image = (image - self.image_mean) / self.image_std
        image_bc = torch.tensor(image_bc, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        image_bc = F.interpolate(image_bc.unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0)

        image_m = torch.tensor(image_m, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        image_m = F.interpolate(image_m.unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0)

        ghi_value = torch.tensor(ghi_value, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image_bc = self.transform(image_bc)
            image_m = self.transform(image_m)


        return (image_bc, image_m), ghi_value  # Return the images and its corresponding GHI value
