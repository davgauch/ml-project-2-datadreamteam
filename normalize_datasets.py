import numpy as np
import cv2  # OpenCV for resizing

def normalize_and_save_in_batches(
    images_path, 
    ghi_values_path, 
    output_images_path, 
    output_ghi_values_path, 
    batch_size=32, 
    new_size=(224, 224), 
    image_mean=None, 
    image_std=None, 
    ghi_min=None, 
    ghi_max=None
):
    """
    Normalize images and GHI values in batches, resize images to new dimensions, and save them to .npy files.

    Args:
        images_path (str): Path to the raw images (.npy file).
        ghi_values_path (str): Path to the raw GHI values (.npy file).
        output_images_path (str): Output path for normalized images.
        output_ghi_values_path (str): Output path for normalized GHI values.
        batch_size (int): Batch size for processing.
        new_size (tuple): Target dimensions for resized images (height, width).
        image_mean (np.array, optional): Precomputed mean for image normalization.
        image_std (np.array, optional): Precomputed std deviation for image normalization.
        ghi_min (float, optional): Minimum GHI value for normalization.
        ghi_max (float, optional): Maximum GHI value for normalization.

    Returns:
        tuple: (image_mean, image_std, ghi_min, ghi_max)
    """
    # Load raw images and GHI values
    images = np.load(images_path, mmap_mode='r')  # Shape: (num_samples, height, width, channels)
    ghi_values = np.load(ghi_values_path)  # Shape: (num_samples,)

    # Remove invalid entries (NaN values in GHI)
    valid_indices = ~np.isnan(ghi_values)
    images = images[valid_indices]
    ghi_values = ghi_values[valid_indices]

    # Determine the number of samples
    num_samples = images.shape[0]

    # Compute normalization statistics if not provided
    if image_mean is None or image_std is None:
        image_mean = images.mean(axis=(0, 1, 2)) / 255.0
        image_std = images.std(axis=(0, 1, 2)) / 255.0

    if ghi_min is None or ghi_max is None:
        ghi_min = ghi_values.min()
        ghi_max = ghi_values.max()

    # Prepare containers for normalized data
    all_images_resized = []
    all_ghi_values = []

    # Process data in batches
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)

        # Batch slicing
        batch_images = images[start_idx:end_idx]
        batch_ghi_values = ghi_values[start_idx:end_idx]

        # Resize and normalize images
        batch_images_resized = np.array([cv2.resize(img, new_size) for img in batch_images])
        batch_images_resized = batch_images_resized / 255.0  # Scale to [0, 1]
        batch_images_resized = (batch_images_resized - image_mean) / image_std  # Normalize

        # Normalize GHI values
        batch_ghi_values = (batch_ghi_values - ghi_min) / (ghi_max - ghi_min)

        # Append processed data
        all_images_resized.append(batch_images_resized)
        all_ghi_values.append(batch_ghi_values)

        # Display progress
        progress = (end_idx / num_samples) * 100
        print(f"Processed {end_idx}/{num_samples} samples ({progress:.2f}%)")

    # Combine batches into single arrays
    all_images_resized = np.vstack(all_images_resized)
    all_ghi_values = np.concatenate(all_ghi_values)

    # Save processed data
    np.save(output_images_path, all_images_resized)
    np.save(output_ghi_values_path, all_ghi_values)

    print(f"Normalized and resized images saved to {output_images_path}")
    print(f"Normalized GHI values saved to {output_ghi_values_path}")

    return image_mean, image_std, ghi_min, ghi_max


def preprocess_and_save(
    training_images_path, 
    training_ghi_values_path, 
    validation_images_path, 
    validation_ghi_values_path, 
    test_images_path, 
    test_ghi_values_path, 
    output_images_path, 
    output_ghi_values_path, 
    batch_size=32, 
    new_size=(224, 224)
):
    """
    Preprocess and save datasets (training, validation, test) with normalization.

    Args:
        training_images_path (str): Path to raw training images.
        training_ghi_values_path (str): Path to raw training GHI values.
        validation_images_path (str): Path to raw validation images.
        validation_ghi_values_path (str): Path to raw validation GHI values.
        test_images_path (str): Path to raw test images.
        test_ghi_values_path (str): Path to raw test GHI values.
        output_images_path (str): Base path for normalized images.
        output_ghi_values_path (str): Base path for normalized GHI values.
        batch_size (int): Batch size for processing.
        new_size (tuple): Target dimensions for resizing images.
    """
    print("Processing training data...")
    image_mean, image_std, ghi_min, ghi_max = normalize_and_save_in_batches(
        training_images_path, training_ghi_values_path,
        f"{output_images_path}_train.npy", f"{output_ghi_values_path}_train.npy",
        batch_size=batch_size, new_size=new_size
    )
    print(f"Image mean: {image_mean}, Image std: {image_std}")
    print(f"GHI min: {ghi_min}, GHI max: {ghi_max}")

    print("Processing validation data...")
    normalize_and_save_in_batches(
        validation_images_path, validation_ghi_values_path,
        f"{output_images_path}_val.npy", f"{output_ghi_values_path}_val.npy",
        batch_size=batch_size, new_size=new_size, 
        image_mean=image_mean, image_std=image_std, 
        ghi_min=ghi_min, ghi_max=ghi_max
    )

    print("Processing test data...")
    normalize_and_save_in_batches(
        test_images_path, test_ghi_values_path,
        f"{output_images_path}_test.npy", f"{output_ghi_values_path}_test.npy",
        batch_size=batch_size, new_size=new_size, 
        image_mean=image_mean, image_std=image_std, 
        ghi_min=ghi_min, ghi_max=ghi_max
    )


if __name__ == "__main__":
    # Paths for raw and processed datasets
    training_images_path = "./data/X_BC_train.npy"
    training_ghi_values_path = "./data/labels_train.npy"
    validation_images_path = "./data/X_BC_val.npy"
    validation_ghi_values_path = "./data/labels_val.npy"
    test_images_path = "./data/X_BC_test.npy"
    test_ghi_values_path = "./data/labels_test.npy"

    output_images_path = "./data/normalized_X_BC"
    output_ghi_values_path = "./data/normalized_labels"

    # Preprocess and save
    preprocess_and_save(
        training_images_path, training_ghi_values_path,
        validation_images_path, validation_ghi_values_path,
        test_images_path, test_ghi_values_path,
        output_images_path, output_ghi_values_path,
        batch_size=32, new_size=(224, 224)
    )
