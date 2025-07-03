import cv2
import numpy as np
from sklearn.cluster import KMeans
import joblib
import os
import glob
import time

# --- Configuration ---
# !!! IMPORTANT: SET THIS PATH TO YOUR ORIGINAL TRAINING IMAGE DATASET !!!
# This should be the dataset used to train Sift_model.h5, SiftAHE_model.h5, etc.
# TRAINING_IMAGE_DIR = 'path/to/your/ORIGINAL/training/images' # Old single path
TRAINING_IMAGE_DIRS = [ # Changed to a list of directories
    r'e:\bcrrp\Output_Files\SIFT_Images',
    r'e:\bcrrp\Output_Files\SiftAHE_Images',
    r'e:\bcrrp\Output_Files\SiftNeg_Images'
]

# Parameters matching the notebook and inference_app.py
N_CLUSTERS = 100 # Vocabulary size (matches notebook)
FEATURE_IMG_SIZE = (224, 224) # Image size used for feature extraction
MODEL_DIR = 'e:\\bcrrp\\models'
KMEANS_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'sift_kmeans_model.joblib')

# --- Feature Extraction Function ---
def extract_sift_descriptors(image_path):
    """Extracts SIFT descriptors from a single image."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        # Resize to the standard size used in training/inference
        img_resized = cv2.resize(img, FEATURE_IMG_SIZE)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img_resized, None)
        return des
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- Main Training Logic ---
if __name__ == "__main__":
    start_time = time.time()
    print("--- Starting SIFT KMeans Model Creation ---")

    # Check if any default paths remain (optional, but good practice)
    if any('path/to/your/ORIGINAL/training/images' in d for d in TRAINING_IMAGE_DIRS):
         print("\n" + "="*60)
         print("ERROR: Please ensure all paths in 'TRAINING_IMAGE_DIRS' are updated")
         print("       to point to your actual training image datasets.")
         print("="*60 + "\n")
         exit() # Stop execution if default path is found

    print(f"\n1. Searching for training images in specified directories:")
    image_paths = []
    # Iterate through each directory in the list
    for train_dir in TRAINING_IMAGE_DIRS:
        print(f"   - Searching in: {train_dir}")
        if not os.path.isdir(train_dir):
            print(f"     Warning: Directory not found: {train_dir}. Skipping.")
            continue
        # Search recursively for common image types within the current directory
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
             image_paths.extend(glob.glob(os.path.join(train_dir, '**', ext), recursive=True))

    if not image_paths:
        print(f"\nError: No images found in any of the specified directories. Please check the paths in TRAINING_IMAGE_DIRS and file types (png, jpg, bmp, tif).")
        exit()
    else:
        print(f"\n   Found {len(image_paths)} potential training images across all directories.")

    print("\n2. Extracting SIFT descriptors from training images...")
    all_descriptors = []
    processed_count = 0
    extraction_start_time = time.time()

    for i, img_path in enumerate(image_paths):
        descriptors = extract_sift_descriptors(img_path)
        if descriptors is not None and len(descriptors) > 0:
            all_descriptors.append(descriptors)
            processed_count += 1
        # Print progress periodically
        if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
             elapsed = time.time() - extraction_start_time
             print(f"   Processed {i + 1}/{len(image_paths)} images... ({processed_count} yielded descriptors) [{elapsed:.1f}s]")

    if not all_descriptors:
        print("\nError: No SIFT descriptors were successfully extracted from any images. Cannot train KMeans.")
        exit()
    else:
        print(f"\n   Successfully extracted descriptors from {processed_count} images.")

    # Stack all descriptors into a single numpy array
    print("\n3. Stacking descriptors into a single array...")
    try:
        stacked_descriptors = np.vstack(all_descriptors)
        print(f"   Total descriptors collected: {stacked_descriptors.shape[0]}")
        # Optional: Check memory usage if it's very large
        # print(f"   Approximate memory usage: {stacked_descriptors.nbytes / (1024**2):.2f} MB")
    except MemoryError:
         print("\nError: Ran out of memory while stacking descriptors.")
         print("Consider processing images in batches or using a machine with more RAM.")
         exit()
    except Exception as e:
         print(f"\nError stacking descriptors: {e}")
         exit()


    # --- Train KMeans ---
    print(f"\n4. Training KMeans with {N_CLUSTERS} clusters...")
    print(f"   This may take some time depending on the number of descriptors ({stacked_descriptors.shape[0]})...")
    kmeans_start_time = time.time()
    # Using n_init=10 (default in newer sklearn) to run KMeans multiple times with different seeds
    # and choose the best one, which helps avoid poor local minima. It also suppresses a future warning.
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, verbose=0) # Set verbose=1 for progress
    try:
        kmeans.fit(stacked_descriptors)
        kmeans_elapsed = time.time() - kmeans_start_time
        print(f"   KMeans training complete. [{kmeans_elapsed:.1f}s]")
    except MemoryError:
         print("\nError: Ran out of memory during KMeans training.")
         print("Consider reducing the number of descriptors (sampling), reducing N_CLUSTERS,")
         print("or using MiniBatchKMeans for larger datasets.")
         exit()
    except Exception as e:
         print(f"\nError during KMeans training: {e}")
         exit()

    # --- Save the Model ---
    print(f"\n5. Saving trained KMeans model to: {KMEANS_MODEL_SAVE_PATH}")
    try:
        os.makedirs(os.path.dirname(KMEANS_MODEL_SAVE_PATH), exist_ok=True) # Ensure directory exists
        joblib.dump(kmeans, KMEANS_MODEL_SAVE_PATH)
        print("   KMeans model saved successfully.")
    except Exception as e:
        print(f"\nError saving KMeans model: {e}")
        exit()

    total_elapsed = time.time() - start_time
    print(f"\n--- Finished KMeans Model Creation ---")
    print(f"Total time: {total_elapsed:.2f} seconds")