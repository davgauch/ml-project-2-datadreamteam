import numpy as np
import cv2  # OpenCV for resizing

def normalize_and_save_in_batches(images_path, ghi_values_path, output_images_path, output_ghi_values_path, batch_size=32, new_size=(224, 224), image_mean=None, image_std=None, ghi_min=None, ghi_max=None, normalize_labels=True):
    """
    Normalize images and GHI values in batches, resize images to new dimensions, and save them to .npy files.
    
    Args:
        images_path (str): Path to input images (numpy .npy file).
        ghi_values_path (str): Path to GHI values (numpy .npy file).
        output_images_path (str): Path to save normalized images.
        output_ghi_values_path (str): Path to save normalized GHI values.
        batch_size (int): Number of samples to process in each batch.
        new_size (tuple): Target size for resizing images.
        image_mean (array-like): Mean for image normalization (optional).
        image_std (array-like): Standard deviation for image normalization (optional).
        ghi_min (float): Minimum GHI value for normalization (optional).
        ghi_max (float): Maximum GHI value for normalization (optional).
        normalize_labels (bool): Whether to normalize GHI values (default: True).
    
    Returns:
        tuple: Computed or provided (image_mean, image_std, ghi_min, ghi_max).
    """
    # Load the raw images and GHI values
    images = np.load(images_path, mmap_mode='r')  # Shape: (num_samples, height, width, channels)
    ghi_values = np.load(ghi_values_path)  # Shape: (num_samples,)

    # Remove rows where GHI values are NaN
    valid_indices = ~np.isnan(ghi_values)
    images = images[valid_indices]
    ghi_values = ghi_values[valid_indices]

    # Number of samples
    num_samples = images.shape[0]
    
    # Compute normalization statistics if not provided (for this dataset)
    if image_mean is None or image_std is None:
        image_mean = images.mean(axis=(0, 1, 2)) / 255.0
        image_std = images.std(axis=(0, 1, 2)) / 255.0

    if ghi_min is None or ghi_max is None:
        ghi_min = ghi_values.min()
        ghi_max = ghi_values.max()

    # Initialize lists to store the processed images and GHI values
    all_images_resized = []
    all_ghi_values = []

    # Process images in batches
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)

        # Get the batch
        batch_images = images[start_idx:end_idx]
        batch_ghi_values = ghi_values[start_idx:end_idx]

        # Resize images to the new size (224x224)
        batch_images_resized = np.array([cv2.resize(img, new_size) for img in batch_images])

        # Normalize the images (Scale pixel values from [0, 255] to [0, 1])
        batch_images_resized = batch_images_resized / 255.0
        # Normalize using the mean and std computed for this dataset
        batch_images_resized = (batch_images_resized - image_mean) / image_std

        if normalize_labels:
            # Normalize GHI values (between 0 and 1) using the min/max for this dataset
            batch_ghi_values = (batch_ghi_values - ghi_min) / (ghi_max - ghi_min)

        # Append the processed batch to the list
        all_images_resized.append(batch_images_resized)
        all_ghi_values.append(batch_ghi_values)

        # Print progress
        progress = (end_idx / num_samples) * 100
        print(f"Processed {end_idx}/{num_samples} samples ({progress:.2f}%)", flush=True)

    # Combine all batches into single arrays
    all_images_resized = np.vstack(all_images_resized)
    all_ghi_values = np.concatenate(all_ghi_values)

    # Save the processed images and GHI values to .npy files
    np.save(output_images_path, all_images_resized)
    np.save(output_ghi_values_path, all_ghi_values)

    print(f"Normalized and resized images saved to {output_images_path}")
    print(f"Normalized GHI values saved to {output_ghi_values_path}")
    
    return image_mean, image_std, ghi_min, ghi_max


def preprocess_and_save(training_images_path, training_ghi_values_path, validation_images_path, validation_ghi_values_path, test_images_path, test_ghi_values_path, output_images_path, output_ghi_values_path, batch_size=32, new_size=(224, 224), normalize_labels=True):
    """
    Preprocess and save training, validation, and test data to .npy files.
    
    Args:
        Same as `normalize_and_save_in_batches`, but processes each dataset (training, validation, test) separately.
    """
    # Normalize and save training data and get normalization parameters
    print("Processing training data...")
    image_mean, image_std, ghi_min, ghi_max = normalize_and_save_in_batches(
        training_images_path, training_ghi_values_path, f"{output_images_path}_train.npy", f"{output_ghi_values_path}_train.npy",
        batch_size=batch_size, new_size=new_size, normalize_labels=normalize_labels
    )

    print(f"Image mean: {image_mean}")
    print(f"Image std: {image_std}")
    print(f"GHI min: {ghi_min}")
    print(f"GHI max: {ghi_max}")

    # Preprocess and save validation data (using stats from training set)
    print("Processing validation data...")
    normalize_and_save_in_batches(
        validation_images_path, validation_ghi_values_path, f"{output_images_path}_val.npy", f"{output_ghi_values_path}_val.npy",
        batch_size=batch_size, new_size=new_size, image_mean=image_mean, image_std=image_std, ghi_min=ghi_min, ghi_max=ghi_max, normalize_labels=normalize_labels
    )

    # Preprocess and save test data (using stats from training set)
    print("Processing test data...")
    normalize_and_save_in_batches(
        test_images_path, test_ghi_values_path, f"{output_images_path}_test.npy", f"{output_ghi_values_path}_test.npy",
        batch_size=batch_size, new_size=new_size, image_mean=image_mean, image_std=image_std, ghi_min=ghi_min, ghi_max=ghi_max, normalize_labels=normalize_labels
    )


if __name__ == "__main__":
    # Define paths for training, validation, and test sets
    training_images_path = "./data/X_M_train.npy"  # Path to training images
    training_ghi_values_path = "./data/labels_train.npy"  # Path to training GHI values
    validation_images_path = "./data/X_M_val.npy"  # Path to validation images
    validation_ghi_values_path = "./data/labels_val.npy"  # Path to validation GHI values
    test_images_path = "./data/X_M_test.npy"  # Path to test images
    test_ghi_values_path = "./data/labels_test.npy"  # Path to test GHI values
    
    output_images_path = "./data/normalized_X_M"  # Base path for saving normalized images (all sets)
    output_ghi_values_path = "./data/normalized_labels"  # Base path for saving normalized GHI values (all sets)

    # Preprocess and save all datasets (training, validation, test)
    preprocess_and_save(
        training_images_path, training_ghi_values_path, validation_images_path, validation_ghi_values_path, 
        test_images_path, test_ghi_values_path, output_images_path, output_ghi_values_path, batch_size=32, new_size=(224, 224), normalize_labels=False
    )