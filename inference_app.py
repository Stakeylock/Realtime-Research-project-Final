import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd # Added
import joblib      # Added
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import cv2
import os
from skimage.feature import hog, local_binary_pattern # For HOG/LBP placeholders
from sklearn.exceptions import NotFittedError # Added

# --- Configuration ---
MODEL_DIR = 'e:\\bcrrp\\models'
# Image Model Paths
MODEL_PATHS = {
    'Hog': os.path.join(MODEL_DIR, 'Hog_model.h5'),
    'HogAHE': os.path.join(MODEL_DIR, 'HogAHE_model.h5'),
    'HogN': os.path.join(MODEL_DIR, 'HogN_model.h5'),
    'LBP': os.path.join(MODEL_DIR, 'LBP_model.h5'),
    'LBPAHE': os.path.join(MODEL_DIR, 'LBPAHE_model.h5'),
    'LBPN': os.path.join(MODEL_DIR, 'LBPN_model.h5'),
    'ResNet': os.path.join(MODEL_DIR, 'resnet_model.h5'),
    'ResNetAHE': os.path.join(MODEL_DIR, 'resnetAHE_model.h5'),
    'ResNetN': os.path.join(MODEL_DIR, 'resnetN_model.h5'),
    'Sift': os.path.join(MODEL_DIR, 'Sift_model.h5'),
    'SiftAHE': os.path.join(MODEL_DIR, 'SiftAHE_model.h5'),
    'SiftN': os.path.join(MODEL_DIR, 'SiftN_model.h5'),
}
# Gene Model Paths (Added)
GENE_MODEL_PATH = os.path.join(MODEL_DIR, 'gene_expression_model.joblib')
GENE_SCALER_PATH = os.path.join(MODEL_DIR, 'gene_expression_scaler.joblib')
GENE_FEATURES_PATH = os.path.join(MODEL_DIR, 'gene_feature_names.joblib')
# SIFT KMeans Model Path (Added)
SIFT_KMEANS_PATH = os.path.join(MODEL_DIR, 'sift_kmeans_model.joblib') # Assuming this is the filename

# Image size used for ResNet models
RESNET_IMG_SIZE = (224, 224)
# Image size used for HOG/LBP/SIFT feature extraction (matches notebook)
FEATURE_IMG_SIZE = (224, 224)
# LBP parameters from notebook
LBP_RADIUS = 3
LBP_N_POINTS = 24 # P = 24 in notebook

# --- Global Variables for Loaded Models --- (Ensure these are defined if not already)
sift_kmeans_model = None
# models = {} # Already defined
# gene_pipeline = { ... } # Already defined

# --- Model Loading ---
models = {}
# Initialize gene model pipeline
gene_pipeline = {
    'model': None,
    'scaler': None,
    'features': None
}

def load_gene_model():
    """Load the gene expression model and related components"""
    global gene_pipeline
    try:
        gene_pipeline['model'] = joblib.load(GENE_MODEL_PATH)
        gene_pipeline['scaler'] = joblib.load(GENE_SCALER_PATH)
        gene_pipeline['features'] = joblib.load(GENE_FEATURES_PATH)
        print("Gene expression model loaded successfully.")
        return True
    except FileNotFoundError as e:
        print(f"Error loading gene model: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error loading gene model: {e}")
        return False

# Call this function during initialization
load_gene_model()

def load_all_models():
    """Loads all models specified in MODEL_PATHS and gene/SIFT pipeline components."""
    global sift_kmeans_model # Added global variable for SIFT KMeans
    print("Loading image models...")
    loaded_image_count = 0
    # Load Image Models (.h5)
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                # Suppress TensorFlow loading warnings temporarily if needed
                # tf.get_logger().setLevel('ERROR')
                models[name] = load_model(path, compile=False)
                # tf.get_logger().setLevel('INFO') # Restore logging level
                print(f"Loaded {name} from {path}")
                loaded_image_count += 1
            except Exception as e:
                print(f"Error loading image model {name} from {path}: {e}")
                models[name] = None # Mark as failed
        else:
            print(f"Image model file not found: {path}")
            models[name] = None
    print(f"Finished loading image models. {loaded_image_count}/{len(MODEL_PATHS)} loaded successfully.")

    # Load Gene Pipeline Components (.joblib) - Added
    print("\nLoading gene expression pipeline components...")
    loaded_gene_count = 0
    try:
        if os.path.exists(GENE_MODEL_PATH):
            gene_pipeline['model'] = joblib.load(GENE_MODEL_PATH)
            print(f"Loaded gene model from {GENE_MODEL_PATH}")
            loaded_gene_count += 1
        else:
            print(f"Gene model file not found: {GENE_MODEL_PATH}")

        if os.path.exists(GENE_SCALER_PATH):
            gene_pipeline['scaler'] = joblib.load(GENE_SCALER_PATH)
            print(f"Loaded gene scaler from {GENE_SCALER_PATH}")
            loaded_gene_count += 1
        else:
            print(f"Gene scaler file not found: {GENE_SCALER_PATH}")

        if os.path.exists(GENE_FEATURES_PATH):
            gene_pipeline['features'] = joblib.load(GENE_FEATURES_PATH)
            print(f"Loaded gene feature names from {GENE_FEATURES_PATH}")
            loaded_gene_count += 1
        else:
            print(f"Gene feature names file not found: {GENE_FEATURES_PATH}")

    except Exception as e:
        print(f"Error loading gene pipeline component: {e}")
        # Don't mark individual components as None here, check later

    print(f"Finished loading gene components. {loaded_gene_count}/3 loaded successfully.")

    # Load SIFT KMeans Model (.joblib) - Added
    print("\nLoading SIFT KMeans model...")
    try:
        if os.path.exists(SIFT_KMEANS_PATH):
            sift_kmeans_model = joblib.load(SIFT_KMEANS_PATH)
            print(f"Loaded SIFT KMeans model from {SIFT_KMEANS_PATH}")
        else:
            print(f"SIFT KMeans model file not found: {SIFT_KMEANS_PATH}")
            sift_kmeans_model = None # Ensure it's None if not found
    except Exception as e:
        print(f"Error loading SIFT KMeans model: {e}")
        sift_kmeans_model = None

    # Check if essential components are loaded
    if loaded_image_count == 0:
         messagebox.showerror("Error", "No image models could be loaded. Please check paths and model files.")
         return False # Exit if no image models loaded

    if loaded_gene_count < 3:
        messagebox.showwarning("Warning", "Could not load all gene pipeline components. Gene prediction will be disabled.")

    if sift_kmeans_model is None:
         messagebox.showwarning("Warning", "Could not load SIFT KMeans model. SIFT predictions will be disabled.")
         # Allow app to run, but SIFT won't work

    return True


# --- Gene Expression Prediction Function ---
def predict_gene_expression_data(gene_data_file_path):
    """
    Predicts outcome based on gene expression data from a file.
    Assumes the file is CSV or TSV with gene names as columns and a single row of data.
    """
    if not all(gene_pipeline.values()):
        return "Error: Gene model, scaler, or feature names not loaded.", None, "Gene pipeline components are missing. Please check server logs."

    model = gene_pipeline['model']
    scaler = gene_pipeline['scaler']
    expected_features = gene_pipeline['features']

    if model is None or scaler is None or expected_features is None:
        return "Error: Essential gene pipeline components are None.", None, "Gene pipeline components are missing critical parts."

    try:
        # Try reading as CSV, then TSV
        try:
            new_data_df = pd.read_csv(gene_data_file_path)
        except pd.errors.ParserError:
            new_data_df = pd.read_csv(gene_data_file_path, sep='\t')
        
        if new_data_df.empty:
            return "Error: Uploaded gene data file is empty or could not be parsed.", None, "File parsing failed."

        # Assuming the first row contains the data, and columns are gene names
        # If multiple rows, take the first one.
        new_data_df = new_data_df.head(1)

        # Align columns with the expected features
        # Add missing columns with 0 (or mean/median if appropriate, but 0 is safer if unsure)
        # Remove extra columns
        new_data_aligned = new_data_df.reindex(columns=expected_features, fill_value=0)

        if new_data_aligned.shape[1] != len(expected_features):
            return "Error: Feature mismatch.", None, f"Input data has {new_data_aligned.shape[1]} features after alignment, model expects {len(expected_features)}."

        # Check for NaNs introduced by alignment or original data
        if new_data_aligned.isnull().values.any():
            print("Warning: Gene input data contains NaN values. Filling with 0.")
            new_data_aligned = new_data_aligned.fillna(0)

        # Apply the *same* scaling
        scaled_new_data = scaler.transform(new_data_aligned) # Use transform, not fit_transform!

        # Predict
        prediction = model.predict(scaled_new_data)
        prediction_proba = model.predict_proba(scaled_new_data)

        # Interpret prediction (modify based on your actual labels)
        # This interpretation is based on the example in train_gene_model.py
        predicted_class_label = "Class 1" if prediction[0] == 1 else "Class 0"
        # predicted_class_label = "Condition 1 (e.g., 32 hours)" if prediction[0] == 1 else "Condition 0 (e.g., 6 hours)"
        
        # Return the predicted class label and the probability of the predicted class
        # For binary classification, predict_proba returns [[prob_class_0, prob_class_1]]
        probability_of_predicted_class = prediction_proba[0][prediction[0]]

        return predicted_class_label, float(probability_of_predicted_class), None # No error message

    except NotFittedError:
        return "Error: Model or scaler not fitted.", None, "The gene model's scaler was not fitted. Train the model first."
    except FileNotFoundError: # Should not happen if path is passed correctly
        return "Error: Gene data file not found (internal error).", None, "Internal server error processing file path."
    except Exception as e:
        print(f"Error during gene prediction: {e}")
        return "Error: Prediction failed.", None, f"An unexpected error occurred: {str(e)}"


# --- Preprocessing ---
# !! IMPORTANT: Replace these with your actual preprocessing logic !!

def preprocess_image_resnet(img_path):
    """Preprocesses image for ResNet models."""
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=RESNET_IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Assuming standard ResNet preprocessing (adjust if different)
        img_array = tf.keras.applications.resnet.preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error preprocessing for ResNet: {e}")
        return None

def extract_hog_features(img_path):
    """
    Extracts HOG features and formats them for model input.
    Returns a 3D tensor with shape (1, 224, 224, 3) as expected by the model.
    """
    print("Extracting HOG features...")
    try:
        # Read and resize image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None
        
        # Convert to grayscale for HOG
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, FEATURE_IMG_SIZE)
        
        # Extract HOG features with visualization
        features, hog_image = hog(
            img_resized, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=True, 
            block_norm='L2-Hys'
        )
        
        # Normalize HOG visualization to 0-255 range
        hog_image = (hog_image * 255).astype("uint8")
        
        # Resize HOG image to match model input size
        hog_image_resized = cv2.resize(hog_image, FEATURE_IMG_SIZE)
        
        # Convert to 3-channel image (duplicate grayscale across channels)
        hog_3channel = np.stack([hog_image_resized] * 3, axis=-1)
        
        # Add batch dimension
        hog_3channel = np.expand_dims(hog_3channel, axis=0)
        
        print(f"HOG features extracted with shape: {hog_3channel.shape}")
        return hog_3channel
    except Exception as e:
        print(f"Error extracting HOG features: {e}")
        return None

def extract_lbp_features(img_path):
    """
    Extracts LBP features and formats them for model input.
    Returns a 3D tensor with shape (1, 224, 224, 3) as expected by the model.
    """
    print("Extracting LBP features...")
    try:
        # Read and resize image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None
        
        # Convert to grayscale for LBP
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, FEATURE_IMG_SIZE)
        
        # Extract LBP features
        lbp = local_binary_pattern(img_resized, LBP_N_POINTS, LBP_RADIUS, method='uniform')
        
        # Normalize LBP to 0-255 range for visualization
        lbp_normalized = np.uint8((lbp / lbp.max()) * 255)
        
        # Convert to 3-channel image (duplicate grayscale across channels)
        lbp_3channel = np.stack([lbp_normalized] * 3, axis=-1)
        
        # Add batch dimension
        lbp_3channel = np.expand_dims(lbp_3channel, axis=0)
        
        print(f"LBP features extracted with shape: {lbp_3channel.shape}")
        return lbp_3channel
    except Exception as e:
        print(f"Error extracting LBP features: {e}")
        return None

def extract_sift_features(img_path):
    """
    Extracts SIFT features, creates a Bag of Visual Words representation,
    and formats the output for model input.
    Returns a 3D tensor with shape (1, 224, 224, 3) as expected by the model.
    """
    global sift_kmeans_model
    
    print("Extracting SIFT features...")
    try:
        if sift_kmeans_model is None:
            print("Error: SIFT KMeans model not loaded. Cannot extract SIFT features.")
            return None
            
        # Read and resize image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None
            
        # Convert to grayscale for SIFT
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, FEATURE_IMG_SIZE)
        
        # Extract SIFT descriptors
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img_resized, None)
        
        if descriptors is None or len(keypoints) == 0:
            print("Warning: No SIFT keypoints found in the image.")
            # Create a blank visualization
            sift_visualization = np.zeros((*FEATURE_IMG_SIZE, 3), dtype=np.uint8)
            sift_visualization = np.expand_dims(sift_visualization, axis=0)
            return sift_visualization
        
        # Create Bag of Visual Words representation
        # Predict cluster for each descriptor
        visual_words = sift_kmeans_model.predict(descriptors)
        
        # Create histogram of visual words
        histogram = np.zeros(sift_kmeans_model.n_clusters)
        for word in visual_words:
            histogram[word] += 1
            
        # Normalize histogram
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
        
        # Create a visualization of the SIFT keypoints
        # Draw keypoints on the image
        blank = np.zeros((*FEATURE_IMG_SIZE, 3), dtype=np.uint8)
        sift_visualization = cv2.drawKeypoints(img_resized, keypoints, blank, 
                                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Ensure the visualization is the right shape and add batch dimension
        sift_visualization = cv2.resize(sift_visualization, FEATURE_IMG_SIZE)
        sift_visualization = np.expand_dims(sift_visualization, axis=0)
        
        print(f"SIFT features extracted with {len(keypoints)} keypoints")
        print(f"SIFT visualization shape: {sift_visualization.shape}")
        return sift_visualization
        
    except Exception as e:
        print(f"Error extracting SIFT features: {e}")
        return None

def get_grad_cam(input_model, img_array, layer_name):
    """Generates Grad-CAM heatmap."""
    try:
        # For sequential models, we need to ensure the model has been called
        # by running a forward pass before accessing inputs/outputs
        if isinstance(input_model, tf.keras.Sequential):
            # Run a forward pass to initialize the model's input/output tensors
            _ = input_model(img_array)
            
        grad_model = None
        try:
            # Try with input_model.inputs (plural) first
            grad_model = tf.keras.models.Model(
                inputs=input_model.inputs,
                outputs=[input_model.get_layer(layer_name).output, input_model.output]
            )
        except (ValueError, TypeError, AttributeError) as e_plural:
            # Fallback to input_model.input (singular) if the first attempt fails
            print(f"Grad-CAM info for {input_model.name}: Failed with .inputs ({e_plural}). Trying .input.")
            try:
                grad_model = tf.keras.models.Model(
                    inputs=[input_model.input], # Ensure it's a list
                    outputs=[input_model.get_layer(layer_name).output, input_model.output]
                )
            except Exception as e_singular:
                print(f"Error creating Grad-CAM model for {input_model.name} with .input as well: {e_singular}")
                return None
        
        if grad_model is None:
            print(f"Error: Could not create Grad-CAM model for {input_model.name}.")
            return None

        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Cast the image tensor to a float-32 data type
            img_array = tf.cast(img_array, tf.float32)
            # Compute activations of the last conv layer and make the prediction
            last_conv_layer_output, predictions = grad_model(img_array)
            
            # Get the index of the predicted class
            if predictions.shape[-1] > 1:
                # Multi-class case
                pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
            else:
                # Binary case (single output neuron)
                pred_index = 0 if predictions[0][0] < 0.5 else 1 # Assuming 0.5 threshold
                class_channel = predictions # Use the raw output for binary
                
        # Gradient of the output neuron with respect to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Vector of mean intensity of the gradient over a specific feature map channel
        if grads is None: # Add check for None grads
            print(f"Error in Grad-CAM: Grads are None for {input_model.name}. Check model output and class_channel.")
            return None
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by corresponding gradients
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except Exception as e:
        print(f"Error in Grad-CAM generation: {e}")
        return None

def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlays the heatmap on the original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    overlayed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed_img

def find_last_conv_layer(model):
    """Finds the name of the last convolutional layer for Grad-CAM."""
    # First try to find a standard convolutional layer
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            print(f"Found convolutional layer for Grad-CAM: {layer.name}")
            return layer.name
            
    # If no Conv2D layer is found, look for other layers that might work with Grad-CAM
    for layer in reversed(model.layers):
        # Check if the layer has a 4D output (batch_size, height, width, channels)
        if hasattr(layer, 'output_shape') and len(getattr(layer, 'output_shape', [])) == 4:
            print(f"Found alternative layer with 4D output for Grad-CAM: {layer.name}")
            return layer.name
            
    # If still no suitable layer, check for nested models/layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
            # This is a nested model or a layer with sublayers
            nested_layer_name = None
            try:
                nested_layers = layer.layers
                for nested_layer in reversed(nested_layers):
                    if isinstance(nested_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        nested_layer_name = f"{layer.name}/{nested_layer.name}"
                        print(f"Found nested convolutional layer for Grad-CAM: {nested_layer_name}")
                        return nested_layer_name
            except (AttributeError, ValueError) as e:
                print(f"Error inspecting nested layer {layer.name}: {e}")
                
    print("Warning: Could not find a suitable layer for Grad-CAM.")
    return None

def create_visualization_grid(original_img_pil, grad_cam_images_dict, thumbnail_size=(200, 200), padding=10):
    """Creates a more compact grid of the original image and Grad-CAM overlays."""
    
    images_to_display = []
    if original_img_pil:
        images_to_display.append(original_img_pil)
    
    # Add valid Grad-CAM images (which are numpy arrays)
    for model_name, img_data in grad_cam_images_dict.items():
        if isinstance(img_data, np.ndarray):
            try:
                # Convert OpenCV image (BGR) to PIL Image (RGB)
                img_pil = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                images_to_display.append(img_pil)
            except Exception as e:
                print(f"Warning: Could not convert Grad-CAM image for {model_name} to PIL: {e}")
        elif isinstance(img_data, Image.Image): # Should not happen if coming from overlay_heatmap
            images_to_display.append(img_data)
        else:
            print(f"Warning: Skipping unknown Grad-CAM data type for {model_name}: {type(img_data)}")

    num_images = len(images_to_display)

    if num_images == 0:
        print("Warning: No images to display in grid.")
        return Image.new('RGB', thumbnail_size, 'grey') # Return a placeholder
    if num_images == 1:
        # If only one image (e.g., original only, or one failed gradcam), return it directly
        return images_to_display[0].resize(thumbnail_size, Image.LANCZOS)

    # Create a more compact grid layout - horizontal row if 3 or fewer images
    if num_images <= 3:
        cols = num_images
        rows = 1
    else:
        # Otherwise aim for a square-ish grid
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))

    # Calculate grid canvas size including padding
    grid_width = cols * thumbnail_size[0] + (cols + 1) * padding
    grid_height = rows * thumbnail_size[1] + (rows + 1) * padding
    
    # Create a new canvas with a dark blue background (matches mammogram style)
    grid_canvas = Image.new('RGB', (grid_width, grid_height), (0, 0, 40)) # Dark blue background
    
    # Place images in a left-to-right, top-to-bottom order
    current_img_idx = 0
    for r in range(rows):
        for c in range(cols):
            if current_img_idx < num_images:
                img_pil = images_to_display[current_img_idx]
                
                # Resize image to thumbnail
                img_pil_resized = img_pil.resize(thumbnail_size, Image.LANCZOS)
                
                # Calculate position with padding
                x_offset = padding + c * (thumbnail_size[0] + padding)
                y_offset = padding + r * (thumbnail_size[1] + padding)
                
                grid_canvas.paste(img_pil_resized, (x_offset, y_offset))
                current_img_idx += 1
            
    return grid_canvas

def predict_image(img_path):
    """Runs prediction using the ensemble of image models."""
    if not models:
        messagebox.showerror("Error", "Models are not loaded.")
        return None, None, None, None, {} # Return empty dict for predictions

    original_img_pil = Image.open(img_path).convert('RGB')
    original_img_cv = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR) # Keep for overlay

    predictions = {}
    processed_data = {} # Store preprocessed data
    grad_cam_results = {} # Store Grad-CAM results for each model

    # --- Preprocess Image Data ---
    # Generate all necessary inputs
    processed_data['ResNet'] = preprocess_image_resnet(img_path)
    processed_data['HOG'] = extract_hog_features(img_path)
    processed_data['LBP'] = extract_lbp_features(img_path)
    processed_data['SIFT'] = extract_sift_features(img_path)

    # Check if ResNet preprocessing failed (needed for Grad-CAM)
    if processed_data['ResNet'] is None:
         messagebox.showwarning("Warning", "Failed to preprocess image for ResNet/Grad-CAM.")
         # Allow continuing, but Grad-CAM might fail

    # --- Run Predictions ---
    for name, model in models.items():
        if model is None:
            predictions[name] = None # Skip if model failed to load
            continue

        data = None
        # Select appropriate preprocessed data based on model name convention
        if 'ResNet' in name:
            data = processed_data['ResNet']
        elif 'Hog' in name:
            data = processed_data['HOG']
        elif 'LBP' in name:
            data = processed_data['LBP']
        elif 'Sift' in name:
            data = processed_data['SIFT']

        # Check if data is available for this model type
        if data is None:
            print(f"Skipping {name}: No suitable preprocessed data generated or model not loaded.")
            predictions[name] = None
            continue

        try:
            # Predict using the selected data
            pred = model.predict(data)[0] # Assuming batch size 1
            # Assuming output is probability for class 1 (Cancer)
            # Adjust index if needed (e.g., pred[1] or np.argmax(pred))
            # Check if output is single value (binary) or multi-value (softmax)
            if len(pred) == 1:
                predictions[name] = pred[0] # Assume single output is P(Cancer)
            elif len(pred) > 1:
                 # Assuming index 1 corresponds to the 'Cancer' class probability
                 predictions[name] = pred[1]
            else:
                 print(f"Warning: Unexpected prediction output shape for {name}: {pred.shape}")
                 predictions[name] = None # Cannot interpret

            if predictions[name] is not None:
                print(f"{name} prediction: {predictions[name]:.4f}")
                
                # Generate Grad-CAM for this model
                try:
                    if hasattr(model, 'layers') and model.layers:
                        last_conv_layer_name = find_last_conv_layer(model)
                        if last_conv_layer_name:
                            # Check if the layer actually exists in the model
                            try:
                                model.get_layer(last_conv_layer_name) # Verify layer exists
                                heatmap = get_grad_cam(model, data, last_conv_layer_name)
                                if heatmap is not None:
                                    # Resize heatmap to match original image dimensions
                                    heatmap_resized = cv2.resize(heatmap, (original_img_cv.shape[1], original_img_cv.shape[0]))
                                    # Convert heatmap to RGB format
                                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                                    # Overlay heatmap on original image
                                    alpha = 0.4  # Transparency factor
                                    overlay = cv2.addWeighted(original_img_cv, 1 - alpha, heatmap_colored, alpha, 0)
                                    # Store the result
                                    grad_cam_results[name] = overlay
                                    print(f"Generated Grad-CAM for {name}")
                            except Exception as e:
                                print(f"Error generating Grad-CAM for {name}: {e}")
                except Exception as e:
                    print(f"Error during Grad-CAM generation for {name}: {e}")
            else:
                 print(f"{name} prediction could not be interpreted.")

        except ValueError as ve:
             # Catch potential shape mismatches specifically
             print(f"Error during prediction for {name} (Potential Shape Mismatch): {ve}")
             print(f"Model expected input shape: {model.input_shape}, Data shape provided: {data.shape}")
             predictions[name] = None
        except Exception as e:
            # Catch other prediction errors
            print(f"Error during prediction for {name}: {e}")
            predictions[name] = None

    # --- Ensemble Predictions ---
    valid_preds = [p for p in predictions.values() if p is not None]
    if not valid_preds:
        messagebox.showerror("Error", "No models could produce a valid prediction.")
        # Return structure expected by caller
        return original_img_pil, None, "Error: No valid predictions", None, predictions # Return current predictions dict

    ensemble_pred_prob = np.mean(valid_preds)
    final_prediction = "Cancer Detected" if ensemble_pred_prob > 0.5 else "Normal"
    result_text = f"{final_prediction} (Avg Prob: {ensemble_pred_prob:.3f})"
    print(f"\nEnsemble Probability (Avg): {ensemble_pred_prob:.4f}")
    print(f"Final Prediction: {final_prediction}")

    # Create a comprehensive visualization with all predictions and Grad-CAMs
    # Group models by type (HOG, LBP, SIFT, ResNet)
    model_groups = {
        'HOG': ['Hog', 'HogAHE', 'HogN'],
        'LBP': ['LBP', 'LBPAHE', 'LBPN'],
        'SIFT': ['Sift', 'SiftAHE', 'SiftN'],
        'ResNet': ['ResNet', 'ResNetAHE', 'ResNetN']
    }
    
    # Create a detailed results dictionary with predictions and visualization
    detailed_results = {}
    for group, model_names in model_groups.items():
        detailed_results[group] = {}
        for name in model_names:
            if name in predictions and predictions[name] is not None:
                pred_value = predictions[name]
                pred_class = "Cancer" if pred_value > 0.5 else "Normal"
                detailed_results[group][name] = {
                    'probability': pred_value,
                    'prediction': pred_class,
                    'grad_cam': grad_cam_results.get(name)
                }
    
    # Create a visualization grid
    # Get the size of the original image
    h, w = original_img_cv.shape[:2]
    
    # We'll create a 4x3 grid (4 model types x 3 variants)
    grid_h = h * 4  # 4 rows for HOG, LBP, SIFT, ResNet
    grid_w = w * 3  # 3 columns for original, AHE, N variants
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Add each Grad-CAM result to the grid
    row_idx = 0
    for group, models_dict in detailed_results.items():
        col_idx = 0
        for name, info in models_dict.items():
            # Calculate position in grid
            y_start = row_idx * h
            y_end = y_start + h
            x_start = col_idx * w
            x_end = x_start + w
            
            if 'grad_cam' in info and info['grad_cam'] is not None:
                # Add the overlay to the grid
                grid_img[y_start:y_end, x_start:x_end] = info['grad_cam']
            else:
                # If Grad-CAM failed, use the original image with a text overlay
                # indicating the prediction but no Grad-CAM
                grid_img[y_start:y_end, x_start:x_end] = original_img_cv.copy()
                cv2.putText(grid_img[y_start:y_end, x_start:x_end], 
                           "No Grad-CAM", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add model name and prediction as text
            text = f"{name}: {info['prediction']} ({info['probability']:.3f})"
            cv2.putText(grid_img[y_start:y_end, x_start:x_end], text, (x_start + 10 - x_start, y_start + 30 - y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            col_idx += 1
        row_idx += 1
    
    # Convert the grid to PIL format
    grid_img_pil = Image.fromarray(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
    
    # Return the original image, the grid of visualizations, the result text, and the ensemble probability
    return original_img_pil, grid_img_pil, result_text, ensemble_pred_prob, predictions


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Detection - Comprehensive Analysis")
        self.root.geometry("1200x900")  # Increased size to accommodate all visualizations

        # Frame for controls
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)

        self.load_button = tk.Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Added Gene Load Button
        self.load_gene_button = tk.Button(self.control_frame, text="Load Gene Data (Optional)", command=self.load_gene_data)
        self.load_gene_button.pack(side=tk.LEFT, padx=5)

        self.predict_button = tk.Button(self.control_frame, text="Predict", command=self.run_prediction, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        # Added Label for Gene File Path
        self.gene_file_label = tk.Label(root, text="No gene data loaded.", font=("Arial", 10))
        self.gene_file_label.pack(pady=2)

        self.result_label = tk.Label(root, text="Load an image and click Predict", font=("Arial", 14), justify=tk.LEFT)
        self.result_label.pack(pady=10)

        # Frame for images
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Configure grid layout
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=3)  # Give more space to visualization grid
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.rowconfigure(1, weight=10)

        # Original image
        tk.Label(self.image_frame, text="Original Image").grid(row=0, column=0, pady=2)
        self.input_canvas = tk.Canvas(self.image_frame, bg='lightgrey')
        self.input_canvas.grid(row=1, column=0, sticky="nsew", padx=5)

        # Comprehensive visualization grid
        tk.Label(self.image_frame, text="All Model Predictions with Grad-CAM").grid(row=0, column=1, pady=2)
        self.output_canvas = tk.Canvas(self.image_frame, bg='lightgrey')
        self.output_canvas.grid(row=1, column=1, sticky="nsew", padx=5)

        # Add a scrollbar for the visualization grid
        self.scrollbar_y = tk.Scrollbar(self.image_frame, orient="vertical", command=self.output_canvas.yview)
        self.scrollbar_y.grid(row=1, column=2, sticky="ns")
        self.output_canvas.configure(yscrollcommand=self.scrollbar_y.set)

        self.scrollbar_x = tk.Scrollbar(self.image_frame, orient="horizontal", command=self.output_canvas.xview)
        self.scrollbar_x.grid(row=2, column=1, sticky="ew")
        self.output_canvas.configure(xscrollcommand=self.scrollbar_x.set)

        # Initialize variables
        self.input_img_path = None
        self.gene_data_path = None
        self.input_img_display = None
        self.output_img_display = None

        # Bind resize event
        self.input_canvas.bind("<Configure>", self.resize_image)
        self.output_canvas.bind("<Configure>", self.resize_image)

    def run_prediction(self):
        """Runs prediction on the loaded image."""
        if not self.input_img_path:
            messagebox.showerror("Error", "Please load an image first.")
            return

        # Clear previous results
        self.result_label.config(text="Running prediction...")
        # Clear canvases if they exist and have content
        if hasattr(self, 'input_canvas_image_id') and self.input_canvas_image_id:
            self.input_canvas.delete(self.input_canvas_image_id)
            self.input_canvas_image_id = None
        if hasattr(self, 'output_canvas_image_id') and self.output_canvas_image_id:
            self.output_canvas.delete(self.output_canvas_image_id)
            self.output_canvas_image_id = None
        if hasattr(self, 'predictions_text_id') and self.predictions_text_id:
            self.predictions_canvas.delete(self.predictions_text_id)
            self.predictions_text_id = None
            # Clear the canvas background as well
            self.predictions_canvas.create_rectangle(0, 0, self.predictions_canvas.winfo_width(), self.predictions_canvas.winfo_height(), fill="white", outline="white")

        self.root.update()  # Update UI to show "Running prediction..."

        try:
            # Run prediction
            # Correctly unpack all 5 values returned by predict_image
            original_img_pil, grid_img_pil, result_text, ensemble_pred_prob, predictions_dict = predict_image(self.input_img_path)
            
            # Update result text
            self.result_label.config(text=result_text)
            
            # Display original image
            if original_img:
                self.display_image(original_img, self.input_canvas, keep_aspect=True)
            
            # Display visualization grid
            if visualization_grid:
                self.display_image(visualization_grid, self.output_canvas, keep_aspect=True)
            
            # Display individual model predictions (New)
            if model_predictions:
                self.display_model_predictions(model_predictions, ensemble_prob)

        except Exception as e:
            print(f"Prediction exception: {e}")
            self.result_label.config(text=f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")


    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.input_img_path = path
            self.display_image(self.input_canvas, self.input_img_path)
            # Clear previous output
            self.output_canvas.delete("all")
            self.output_img_display = None
            # Don't reset gene path here, allow separate loading
            # self.gene_data_path = None
            # self.gene_file_label.config(text="No gene data loaded.")
            self.result_label.config(text="Image loaded. Click Predict.")
            self.predict_button.config(state=tk.NORMAL) # Enable predict once image is loaded

    # Added Method to Load Gene Data
    def load_gene_data(self):
        # Expecting a single row CSV or TSV with gene names as columns
        path = filedialog.askopenfilename(filetypes=[("Gene Data", "*.csv *.tsv *.txt")])
        if path:
            if not all(gene_pipeline.values()):
                 messagebox.showwarning("Warning", "Gene pipeline components not loaded. Cannot use gene data.")
                 self.gene_data_path = None
                 self.gene_file_label.config(text="Gene components missing.")
            else:
                self.gene_data_path = path
                self.gene_file_label.config(text=f"Gene Data: {os.path.basename(path)}")
                # Update result label if prediction already ran
                if "Result:" in self.result_label.cget("text"):
                     self.result_label.config(text="Gene data loaded. Click Predict again to include.")
        else:
             self.gene_data_path = None # Clear if dialog cancelled
             self.gene_file_label.config(text="No gene data loaded.")


    def display_image(self, canvas, img_path_or_pil):
        canvas.delete("all")
        try:
            if isinstance(img_path_or_pil, str):
                img = Image.open(img_path_or_pil)
            else:
                img = img_path_or_pil

            # Store original PIL image for resizing
            if canvas == self.input_canvas:
                self.input_img_pil = img
            elif canvas == self.output_canvas:
                self.output_img_pil = img

            self.resize_image(event=None, target_canvas=canvas) # Initial display

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or display image: {e}")
            if canvas == self.input_canvas:
                self.input_img_path = None
                self.predict_button.config(state=tk.DISABLED)


    def resize_image(self, event, target_canvas=None):
        # Determine which canvas triggered or is targeted
        canvas = target_canvas if target_canvas else event.widget
        img_pil = None

        if canvas == self.input_canvas and hasattr(self, 'input_img_pil'):
            img_pil = self.input_img_pil
        elif canvas == self.output_canvas and hasattr(self, 'output_img_pil'):
            img_pil = self.output_img_pil

        if img_pil is None:
            return # No image loaded for this canvas

        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1: # Avoid division by zero if canvas not ready
            return

        # Calculate aspect ratio
        img_aspect = img_pil.width / img_pil.height
        canvas_aspect = canvas_width / canvas_height

        # Calculate new size to fit canvas while maintaining aspect ratio
        if img_aspect > canvas_aspect:
            # Wider than canvas aspect ratio -> fit width
            new_width = canvas_width
            new_height = int(new_width / img_aspect)
        else:
            # Taller than canvas aspect ratio -> fit height
            new_height = canvas_height
            new_width = int(new_height * img_aspect)

        # Resize the image
        img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        # Display on canvas
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=img_tk)

        # Keep a reference to avoid garbage collection
        if canvas == self.input_canvas:
            self.input_img_display = img_tk
        elif canvas == self.output_canvas:
            self.output_img_display = img_tk


    def run_prediction(self):
        if not self.input_img_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        self.result_label.config(text="Predicting...")
        self.root.update_idletasks() # Update GUI to show "Predicting..."

        try:
            # Correctly unpack all 5 values returned by predict_image
            original_img_pil, grid_img_pil, result_text, ensemble_pred_prob, predictions_dict = predict_image(self.input_img_path)

            if result_text: # Check if prediction was successful
                 self.display_image(self.input_canvas, original_img_pil) # Use original_img_pil for input display
                 self.display_image(self.output_canvas, grid_img_pil)    # Use grid_img_pil for output display
                 self.result_label.config(text=f"Result: {result_text}")
                 # You can now also use ensemble_pred_prob and predictions_dict if needed for the GUI
            else:
                 # Error occurred during prediction, message already shown by predict_image
                 self.result_label.config(text="Prediction failed. Check console for errors.")
                 self.output_canvas.delete("all") # Clear output canvas on error

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")
            self.result_label.config(text="Prediction failed.")
            print(f"Prediction exception: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    if not load_all_models():
        print("Exiting due to model loading failure.")
    else:
        root = tk.Tk()
        app = App(root)
        root.mainloop()