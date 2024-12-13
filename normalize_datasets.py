import numpy as np
import cv2
import pandas as pd
import os

def normalize_and_save_in_batches(images_path, ghi_values_path, output_images_path, output_ghi_values_path,
                                  batch_size=32, new_size=(224, 224), image_mean=None, image_std=None,
                                  ghi_min=None, ghi_max=None, normalize_labels=True):
    images = np.load(images_path, mmap_mode='r')
    ghi_values = np.load(ghi_values_path)

    print(f"[normalize_and_save_in_batches] Initial images shape: {images.shape}")
    print(f"[normalize_and_save_in_batches] Initial ghi_values shape: {ghi_values.shape}")

    num_samples = images.shape[0]

    if image_mean is None or image_std is None:
        image_mean = images.mean(axis=(0, 1, 2)) / 255.0
        image_std = images.std(axis=(0, 1, 2)) / 255.0
    print("[normalize_and_save_in_batches] Computed image_mean and image_std")

    if ghi_min is None or ghi_max is None:
        ghi_min = ghi_values.min()
        ghi_max = ghi_values.max()
    print("[normalize_and_save_in_batches] GHI min:", ghi_min, "GHI max:", ghi_max)

    all_images_resized = []
    all_ghi_values = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)

        batch_images = images[start_idx:end_idx]
        batch_ghi_values = ghi_values[start_idx:end_idx]

        batch_images_resized = np.array([cv2.resize(img, new_size) for img in batch_images])
        batch_images_resized = batch_images_resized / 255.0
        batch_images_resized = (batch_images_resized - image_mean) / image_std

        if normalize_labels:
            batch_ghi_values = (batch_ghi_values - ghi_min) / (ghi_max - ghi_min)

        all_images_resized.append(batch_images_resized)
        all_ghi_values.append(batch_ghi_values)

        progress = (end_idx / num_samples) * 100
        print(f"[normalize_and_save_in_batches] Processed {end_idx}/{num_samples} samples ({progress:.2f}%)", flush=True)

    all_images_resized = np.vstack(all_images_resized)
    all_ghi_values = np.concatenate(all_ghi_values)

    print(f"[normalize_and_save_in_batches] Final resized images shape: {all_images_resized.shape}")
    print(f"[normalize_and_save_in_batches] Final GHI shape: {all_ghi_values.shape}")

    np.save(output_images_path, all_images_resized)
    if normalize_labels:
        ghi_output_path = output_images_path.replace(".npy", "_ghi.npy")
        np.save(ghi_output_path, all_ghi_values)
        print(f"[normalize_and_save_in_batches] Normalized GHI values saved to {ghi_output_path}")

    print(f"[normalize_and_save_in_batches] Normalized and resized images saved to {output_images_path}")
    
    return image_mean, image_std, ghi_min, ghi_max


def normalize_meteo_data(train_meteo_path, val_meteo_path, test_meteo_path, 
                         output_train_path, output_val_path, output_test_path):
    print("[normalize_meteo_data] Loading meteo data...")
    train_meteo = np.load(train_meteo_path)
    val_meteo = np.load(val_meteo_path)
    test_meteo = np.load(test_meteo_path)

    print("[normalize_meteo_data] Shapes before normalization:")
    print("Train meteo:", train_meteo.shape)
    print("Val meteo:", val_meteo.shape)
    print("Test meteo:", test_meteo.shape)

    meteo_mean = train_meteo.mean(axis=0)
    meteo_std = train_meteo.std(axis=0) + 1e-8

    train_meteo_norm = (train_meteo - meteo_mean) / meteo_std
    val_meteo_norm = (val_meteo - meteo_mean) / meteo_std
    test_meteo_norm = (test_meteo - meteo_mean) / meteo_std

    np.save(output_train_path, train_meteo_norm)
    np.save(output_val_path, val_meteo_norm)
    np.save(output_test_path, test_meteo_norm)

    print("[normalize_meteo_data] Normalized meteo data saved:")
    print(f"Train: {output_train_path} shape: {train_meteo_norm.shape}")
    print(f"Val:   {output_val_path} shape: {val_meteo_norm.shape}")
    print(f"Test:  {output_test_path} shape: {test_meteo_norm.shape}")

    return meteo_mean, meteo_std


def preprocess_and_save(training_images_path, training_ghi_values_path, 
                        validation_images_path, validation_ghi_values_path, 
                        test_images_path, test_ghi_values_path, 
                        output_images_path, output_ghi_values_path, 
                        batch_size=32, new_size=(224, 224), normalize_labels=True,
                        train_meteo_path=None, val_meteo_path=None, test_meteo_path=None, 
                        output_meteo_train_path=None, output_meteo_val_path=None, output_meteo_test_path=None,
                        train_time_path=None, val_time_path=None, test_time_path=None):
    """
    Preprocess and save training, validation, and test data. Align meteo data with image data by 'Time'.
    """

    def load_images_and_ghi(images_path, ghi_path):
        images = np.load(images_path, mmap_mode='r')
        ghi_values = np.load(ghi_path)
        print(f"[load_images_and_ghi] {images_path} shape: {images.shape}, {ghi_path} shape: {ghi_values.shape}")
        return images, ghi_values

    def load_meteo_csv(meteo_csv_path):
        df = pd.read_csv(meteo_csv_path, low_memory=False)
        # Remove the first column (index column)
        df = df.iloc[:, 1:]
        df["Time"] = pd.to_datetime(df["Time"], format="%d.%m.%Y %H:%M", errors="coerce")
        print(f"[load_meteo_csv] {meteo_csv_path} head:\n", df.head())
        return df

    # Load image arrays and GHI
    train_images, train_ghi = load_images_and_ghi(training_images_path, training_ghi_values_path)
    val_images, val_ghi = load_images_and_ghi(validation_images_path, validation_ghi_values_path)
    test_images, test_ghi = load_images_and_ghi(test_images_path, test_ghi_values_path)

    print("[preprocess_and_save] Initial shapes:")
    print("Train images:", train_images.shape, "Train GHI:", train_ghi.shape)
    print("Val images:", val_images.shape, "Val GHI:", val_ghi.shape)
    print("Test images:", test_images.shape, "Test GHI:", test_ghi.shape)

    train_times = np.load(train_time_path, allow_pickle=True) if train_time_path else None
    val_times = np.load(val_time_path, allow_pickle=True) if val_time_path else None
    test_times = np.load(test_time_path, allow_pickle=True) if test_time_path else None

    if train_times is not None:
        train_time_df = pd.DataFrame({"Time": train_times})
        train_time_df.set_index("Time", inplace=True)
        if train_time_df.index.duplicated().any():
            print("[preprocess_and_save] train_time_df has duplicates!")
            train_time_df = train_time_df[~train_time_df.index.duplicated(keep='first')]
    else:
        train_time_df = None

    if val_times is not None:
        val_time_df = pd.DataFrame({"Time": val_times})
        val_time_df.set_index("Time", inplace=True)
        if val_time_df.index.duplicated().any():
            print("[preprocess_and_save] val_time_df has duplicates!")
            val_time_df = val_time_df[~val_time_df.index.duplicated(keep='first')]
    else:
        val_time_df = None

    if test_times is not None:
        test_time_df = pd.DataFrame({"Time": test_times})
        test_time_df.set_index("Time", inplace=True)
        if test_time_df.index.duplicated().any():
            print("[preprocess_and_save] test_time_df has duplicates!")
            test_time_df = test_time_df[~test_time_df.index.duplicated(keep='first')]
    else:
        test_time_df = None

    if train_meteo_path and val_meteo_path and test_meteo_path:
        train_meteo_df = load_meteo_csv(train_meteo_path)
        val_meteo_df = load_meteo_csv(val_meteo_path)
        test_meteo_df = load_meteo_csv(test_meteo_path)

        train_meteo_df.set_index("Time", inplace=True)
        val_meteo_df.set_index("Time", inplace=True)
        test_meteo_df.set_index("Time", inplace=True)

        # Check duplicates in meteo
        if train_meteo_df.index.duplicated().any():
            print("[preprocess_and_save] train_meteo_df has duplicates!")
            train_meteo_df = train_meteo_df[~train_meteo_df.index.duplicated(keep='first')]
        if val_meteo_df.index.duplicated().any():
            print("[preprocess_and_save] val_meteo_df has duplicates!")
            val_meteo_df = val_meteo_df[~val_meteo_df.index.duplicated(keep='first')]
        if test_meteo_df.index.duplicated().any():
            print("[preprocess_and_save] test_meteo_df has duplicates!")
            test_meteo_df = test_meteo_df[~test_meteo_df.index.duplicated(keep='first')]

        if train_time_df is not None:
            train_meteo_df = train_meteo_df.reindex(index=train_time_df.index, method=None)
        if val_time_df is not None:
            val_meteo_df = val_meteo_df.reindex(index=val_time_df.index, method=None)
        if test_time_df is not None:
            test_meteo_df = test_meteo_df.reindex(index=test_time_df.index, method=None)

        train_meteo_raw = train_meteo_df.values if train_meteo_df is not None else None
        val_meteo_raw = val_meteo_df.values if val_meteo_df is not None else None
        test_meteo_raw = test_meteo_df.values if test_meteo_df is not None else None

        print("[preprocess_and_save] After aligning meteo with time:")
        print("Train meteo:", None if train_meteo_raw is None else train_meteo_raw.shape)
        print("Val meteo:", None if val_meteo_raw is None else val_meteo_raw.shape)
        print("Test meteo:", None if test_meteo_raw is None else test_meteo_raw.shape)

    else:
        train_meteo_raw = val_meteo_raw = test_meteo_raw = None

    def truncate_to_min_length(imgs, ghi, meteo):
        lengths = [len(imgs), len(ghi)]
        if meteo is not None:
            lengths.append(len(meteo))
        min_len = min(lengths)
        print(f"[truncate_to_min_length] min_len: {min_len}")
        imgs = imgs[:min_len]
        ghi = ghi[:min_len]
        if meteo is not None:
            meteo = meteo[:min_len]
        return imgs, ghi, meteo

    print("[preprocess_and_save] Before truncation:")
    print("Train:", train_images.shape, train_ghi.shape, None if train_meteo_raw is None else train_meteo_raw.shape)
    print("Val:", val_images.shape, val_ghi.shape, None if val_meteo_raw is None else val_meteo_raw.shape)
    print("Test:", test_images.shape, test_ghi.shape, None if test_meteo_raw is None else test_meteo_raw.shape)

    train_images, train_ghi, train_meteo_raw = truncate_to_min_length(train_images, train_ghi, train_meteo_raw)
    val_images, val_ghi, val_meteo_raw = truncate_to_min_length(val_images, val_ghi, val_meteo_raw)
    test_images, test_ghi, test_meteo_raw = truncate_to_min_length(test_images, test_ghi, test_meteo_raw)

    print("[preprocess_and_save] After truncation:")
    print("Train:", train_images.shape, train_ghi.shape, None if train_meteo_raw is None else train_meteo_raw.shape)
    print("Val:", val_images.shape, val_ghi.shape, None if val_meteo_raw is None else val_meteo_raw.shape)
    print("Test:", test_images.shape, test_ghi.shape, None if test_meteo_raw is None else test_meteo_raw.shape)

    def clean_data(images, ghi, meteo=None):
        valid_indices = ~np.isnan(ghi)
        valid_indices &= ~np.isnan(images).any(axis=(1,2,3))
        if meteo is not None:
            valid_indices &= ~np.isnan(meteo).any(axis=1)

        images = images[valid_indices]
        ghi = ghi[valid_indices]
        if meteo is not None:
            meteo = meteo[valid_indices]
        return images, ghi, meteo

    print("[preprocess_and_save] Before cleaning:")
    print("Train:", train_images.shape, train_ghi.shape, None if train_meteo_raw is None else train_meteo_raw.shape)
    print("Val:", val_images.shape, val_ghi.shape, None if val_meteo_raw is None else val_meteo_raw.shape)
    print("Test:", test_images.shape, test_ghi.shape, None if test_meteo_raw is None else test_meteo_raw.shape)

    train_images, train_ghi, train_meteo = clean_data(train_images, train_ghi, train_meteo_raw)
    val_images, val_ghi, val_meteo = clean_data(val_images, val_ghi, val_meteo_raw)
    test_images, test_ghi, test_meteo = clean_data(test_images, test_ghi, test_meteo_raw)

    print("[preprocess_and_save] After cleaning:")
    print("Train:", train_images.shape, train_ghi.shape, None if train_meteo is None else train_meteo.shape)
    print("Val:", val_images.shape, val_ghi.shape, None if val_meteo is None else val_meteo.shape)
    print("Test:", test_images.shape, test_ghi.shape, None if test_meteo is None else test_meteo.shape)

    os.makedirs("./temp_cleaned_data", exist_ok=True)
    np.save("./temp_cleaned_data/train_images.npy", train_images)
    np.save("./temp_cleaned_data/train_ghi.npy", train_ghi)
    np.save("./temp_cleaned_data/val_images.npy", val_images)
    np.save("./temp_cleaned_data/val_ghi.npy", val_ghi)
    np.save("./temp_cleaned_data/test_images.npy", test_images)
    np.save("./temp_cleaned_data/test_ghi.npy", test_ghi)

    if train_meteo is not None:
        np.save("./temp_cleaned_data/train_meteo.npy", train_meteo)
        np.save("./temp_cleaned_data/val_meteo.npy", val_meteo)
        np.save("./temp_cleaned_data/test_meteo.npy", test_meteo)

    print("[preprocess_and_save] Shapes before normalization:")
    print("Train:", train_images.shape, train_ghi.shape, None if train_meteo is None else train_meteo.shape)
    print("Val:", val_images.shape, val_ghi.shape, None if val_meteo is None else val_meteo.shape)
    print("Test:", test_images.shape, test_ghi.shape, None if test_meteo is None else test_meteo.shape)

    print("Processing training data...")
    image_mean, image_std, ghi_min, ghi_max = normalize_and_save_in_batches(
        "./temp_cleaned_data/train_images.npy", "./temp_cleaned_data/train_ghi.npy",
        f"{output_images_path}_train.npy", f"{output_ghi_values_path}_train.npy",
        batch_size=batch_size, new_size=new_size, normalize_labels=normalize_labels
    )

    print("[preprocess_and_save] After training normalization:")
    print("Image mean:", image_mean, "Image std:", image_std)
    print("GHI min:", ghi_min, "GHI max:", ghi_max)

    print("Processing validation image and GHI data...")
    normalize_and_save_in_batches(
        "./temp_cleaned_data/val_images.npy", "./temp_cleaned_data/val_ghi.npy",
        f"{output_images_path}_val.npy", f"{output_ghi_values_path}_val.npy",
        batch_size=batch_size, new_size=new_size, image_mean=image_mean, image_std=image_std, ghi_min=ghi_min, ghi_max=ghi_max, normalize_labels=normalize_labels
    )

    print("Processing test image and GHI data...")
    normalize_and_save_in_batches(
        "./temp_cleaned_data/test_images.npy", "./temp_cleaned_data/test_ghi.npy",
        f"{output_images_path}_test.npy", f"{output_ghi_values_path}_test.npy",
        batch_size=batch_size, new_size=new_size, image_mean=image_mean, image_std=image_std, ghi_min=ghi_min, ghi_max=ghi_max, normalize_labels=normalize_labels
    )

    if train_meteo is not None and val_meteo is not None and test_meteo is not None and \
       output_meteo_train_path and output_meteo_val_path and output_meteo_test_path:
        print("Processing meteo data before normalization:")
        print("Train meteo:", train_meteo.shape)
        print("Val meteo:", val_meteo.shape)
        print("Test meteo:", test_meteo.shape)

        np.save("./temp_cleaned_data/train_meteo_cleaned.npy", train_meteo)
        np.save("./temp_cleaned_data/val_meteo_cleaned.npy", val_meteo)
        np.save("./temp_cleaned_data/test_meteo_cleaned.npy", test_meteo)

        meteo_mean, meteo_std = normalize_meteo_data(
            "./temp_cleaned_data/train_meteo_cleaned.npy", "./temp_cleaned_data/val_meteo_cleaned.npy", "./temp_cleaned_data/test_meteo_cleaned.npy",
            output_meteo_train_path, output_meteo_val_path, output_meteo_test_path
        )
        print("[preprocess_and_save] After meteo normalization:")
        print("Meteo mean:", meteo_mean, "Meteo std:", meteo_std)


if __name__ == "__main__":
    training_images_path = "./data/X_M_train.npy"
    training_ghi_values_path = "./data/labels_train.npy"
    validation_images_path = "./data/X_M_val.npy"
    validation_ghi_values_path = "./data/labels_val.npy"
    test_images_path = "./data/X_M_test.npy"
    test_ghi_values_path = "./data/labels_test.npy"

    output_images_path = "./data/normalized_X_M"
    output_ghi_values_path = "./data/normalized_labels"

    train_meteo_path = "./data/meteo_data_train.csv"
    val_meteo_path = "./data/meteo_data_val.csv"
    test_meteo_path = "./data/meteo_data_test.csv"

    output_meteo_train_path = "./data/normalized_meteo_train.npy"
    output_meteo_val_path = "./data/normalized_meteo_val.npy"
    output_meteo_test_path = "./data/normalized_meteo_test.npy"

    train_time_path = "./data/common_time_train.npy"
    val_time_path = "./data/common_time_val.npy"
    test_time_path = "./data/common_time_test.npy"

    preprocess_and_save(
        training_images_path, training_ghi_values_path, 
        validation_images_path, validation_ghi_values_path, 
        test_images_path, test_ghi_values_path, 
        output_images_path, output_ghi_values_path, 
        batch_size=32, new_size=(224, 224), normalize_labels=False,
        train_meteo_path=train_meteo_path, 
        val_meteo_path=val_meteo_path, 
        test_meteo_path=test_meteo_path, 
        output_meteo_train_path=output_meteo_train_path,
        output_meteo_val_path=output_meteo_val_path,
        output_meteo_test_path=output_meteo_test_path,
        train_time_path=train_time_path,
        val_time_path=val_time_path,
        test_time_path=test_time_path
    )
