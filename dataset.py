import numpy as np
import torch
from torch.utils.data import Dataset

class WebcamDataset(Dataset):
    """
    Custom Dataset class for loading webcam images and corresponding GHI (Global Horizontal Irradiance) values.

    Args:
        images_path (str): Path to the `.npy` file containing webcam images. The expected shape is (num_samples, height, width, channels).
        ghi_values_path (str): Path to the `.npy` file containing corresponding GHI values. The expected shape is (num_samples,).
        transform (callable, optional): A function/transform to apply to the images. Typically used for image augmentation, normalization, etc.

    Attributes:
        images (numpy.ndarray): Loaded webcam images.
        ghi_values (numpy.ndarray): Loaded GHI values corresponding to the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, images_path, ghi_values_path, transform=None):
        """
        Initializes the WebcamDataset object by loading the image data and GHI values from the provided file paths.

        Args:
            images_path (str): Path to the `.npy` file containing the webcam images.
            ghi_values_path (str): Path to the `.npy` file containing the corresponding GHI values.
            transform (callable, optional): Optional transformation to apply on the image data (e.g., normalization, augmentation).
        """
        
        # Load the image data and GHI values from .npy files
        self.images = np.load(images_path)  # Shape: (num_samples, height, width, channels)
        self.ghi_values = np.load(ghi_values_path)  # Shape: (num_samples,)
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples (images and corresponding GHI values) in the dataset.
        """
        return self.images.shape[0]  # The number of samples is the number of images

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

        # Convert image to tensor (if it's not already)
        image = torch.tensor(image, dtype=torch.float32)
        ghi_value = torch.tensor(ghi_value, dtype=torch.float32)

        # Permute channels and height/width (tensorflow layout)
        image = image.permute(2, 0, 1)

        # Apply any transformations (if provided)
        if self.transform:
            image = self.transform(image)

        return image, ghi_value  # Return the image and its corresponding GHI value
