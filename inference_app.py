import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd 
import joblib    
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import cv2
import os
from skimage.feature import hog, local_binary_pattern 
from sklearn.exceptions import NotFittedError 
from skimage.util import img_as_ubyte
from PIL import ImageDraw, ImageFont
import io 
import base64 
from skimage import exposure
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score # Added for more metrics

# --- Configuration ---
MODEL_DIR = 'e:\\bcrrp\\models' # Ensure this path is correct for your environment
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
GENE_MODEL_PATH = os.path.join(MODEL_DIR, 'gene_expression_model.joblib')
GENE_SCALER_PATH = os.path.join(MODEL_DIR, 'gene_expression_scaler.joblib')
GENE_FEATURES_PATH = os.path.join(MODEL_DIR, 'gene_feature_names.joblib')
SIFT_KMEANS_PATH = os.path.join(MODEL_DIR, 'sift_kmeans_model.joblib')

# Image size used for ResNet models
RESNET_IMG_SIZE = (224, 224)
# Image size used for HOG/LBP/SIFT feature extraction (matches notebook)
FEATURE_IMG_SIZE = (224, 224)
# LBP parameters from notebook
LBP_RADIUS = 3
LBP_N_POINTS = 24 # P = 24 in notebook

sift_kmeans_model = None
models = {} 
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

def load_all_models():
    """Loads all image and gene expression models."""
    global sift_kmeans_model
    print("Loading image models...")
    loaded_image_count = 0
    # Load Image Models (.h5)
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                models[name] = load_model(path, compile=False)
                print(f"Loaded {name} from {path}")
                loaded_image_count += 1
            except Exception as e:
                print(f"Error loading image model {name} from {path}: {e}")
                models[name] = None 
        else:
            print(f"Image model file not found: {path}")
            models[name] = None
    print(f"Finished loading image models. {loaded_image_count}/{len(MODEL_PATHS)} loaded successfully.")

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
        

    print(f"Finished loading gene components. {loaded_gene_count}/3 loaded successfully.")

    print("\nLoading SIFT KMeans model...")
    try:
        if os.path.exists(SIFT_KMEANS_PATH):
            sift_kmeans_model = joblib.load(SIFT_KMEANS_PATH)
            # Store SIFT KMeans model in the 'models' dictionary for consistency
            models['sift_kmeans_model'] = sift_kmeans_model
            print(f"Loaded SIFT KMeans model from {SIFT_KMEANS_PATH}")
        else:
            print(f"SIFT KMeans model file not found: {SIFT_KMEANS_PATH}")
            sift_kmeans_model = None 
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
         messagebox.showwarning("Warning", "Could not load SIFT KMeans model. SIFT predictions might be disabled or incorrect.")
         # Allow app to run, but SIFT won't work

    return True

def get_label_from_filename(filepath):
    """
    Extracts the ground truth label from the image filename.
    Labels can be 'benign', 'normal', '0' (for class 0) or 'malignant', '1' (for class 1).
    Returns 0 for benign/normal, 1 for malignant, or None if no label is found.
    """
    if not filepath:
        return None
    # Normalize filename for consistent matching
    filename = os.path.basename(filepath).lower()
    
    # Define keywords for each class
    malignant_keywords = ['malignant', '1']
    benign_keywords = ['benign', 'normal', '0']

    if any(keyword in filename for keyword in malignant_keywords):
        return 1
    if any(keyword in filename for keyword in benign_keywords):
        return 0
        
    return None # Return None if no relevant keyword is found

def calculate_validation_metrics(predictions_dict, image_path):
    """
    Calculates performance and trustability metrics by comparing predictions
    to a ground truth label extracted from the filename.

    Args:
        predictions_dict (dict): Dictionary of model predictions, e.g., {'ResNet': (prob, class), ...}
                                 Note: The predictions_dict now contains probabilities directly.
        image_path (str): The path to the image file to check for a label.

    Returns:
        dict: A dictionary containing the found label, performance, and trustability metrics.
    """
    true_label = get_label_from_filename(image_path)
    
    metrics = {
        "true_label_found": true_label is not None,
        "true_label": true_label,
        "performance": {},
        "trustability": {}
    }

    # If no label is found in the filename, we cannot calculate metrics.
    if true_label is None:
        return metrics

    y_true = [int(true_label)]
    
    for model_name, prob in predictions_dict.items():
        if prob is None:
            continue # Skip models that didn't produce a prediction

        # Convert probability to predicted class (0 or 1)
        predicted_class = 1 if prob > 0.5 else 0
        y_pred = [predicted_class]

        # --- Performance Metrics ---
        # Accuracy, Precision, Recall, F1 Score
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        
        metrics["performance"][model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # --- Trustability Metric ---
        # Is the prediction correct?
        is_correct = (predicted_class == true_label)
        trust_score = "High" if is_correct else "Low" # Simple heuristic
        metrics["trustability"][model_name] = {
            'trust_score': trust_score,
            'is_correct': is_correct,
            'predicted_class': predicted_class # Include predicted class for clarity
        }
        
    return metrics


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
        predicted_class_label = "Malignant" if prediction[0] == 1 else "Benign"
        
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

def extract_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None # Return both input and viz
    
    img_resized = cv2.resize(img, FEATURE_IMG_SIZE)
    
    # Calculate HOG features and get the visualization
    fd, hog_image = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualize=True) 
    
    # Rescale HOG visualization to be between 0 and 1, then convert to 8-bit unsigned integer
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1))
    hog_image_display = img_as_ubyte(hog_image_rescaled)
    
    # Convert to 3 channels for consistency if models expect it, and add batch dim
    hog_input_for_model = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR) # Preprocess original for HOG model input
    hog_input_for_model = np.expand_dims(hog_input_for_model, axis=0)
    
    # Convert hog_image_display (grayscale) to 3-channel for visualization grid
    hog_viz_3_channel = cv2.cvtColor(hog_image_display, cv2.COLOR_GRAY2BGR)

    print(f"Extracting HOG features... (input shape: {hog_input_for_model.shape})")
    # Return model input and visualization
    return hog_input_for_model, Image.fromarray(hog_viz_3_channel)

def extract_lbp_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None # Return both input and viz
    
    img_resized = cv2.resize(img, FEATURE_IMG_SIZE) # Resize image before LBP computation

    radius = 3
    n_points = 8 * radius
    lbp_image = local_binary_pattern(img_resized, n_points, radius, method="uniform")
    
    # Normalize LBP image for model input and display (0-255)
    lbp_min = lbp_image.min()
    lbp_max = lbp_image.max()
    if lbp_max - lbp_min == 0: # Handle cases of uniform image
        lbp_image_normalized = np.zeros_like(lbp_image, dtype=np.uint8)
    else:
        lbp_image_normalized = ((lbp_image - lbp_min) / (lbp_max - lbp_min) * 255).astype(np.uint8)

    # Convert normalized LBP image to 3 channels and ensure correct size for model input
    lbp_input_for_model = cv2.cvtColor(lbp_image_normalized, cv2.COLOR_GRAY2BGR)
    lbp_input_for_model = cv2.resize(lbp_input_for_model, FEATURE_IMG_SIZE) # Ensure it's 224x224
    
    # Add batch dimension and convert to float32 as expected by Keras models
    lbp_input_for_model = np.expand_dims(lbp_input_for_model, axis=0).astype(np.float32)

    # Visualization image (already in 3-channel from lbp_input_for_model preparation if needed for display directly)
    # Re-using lbp_image_normalized for a dedicated display variable
    lbp_viz_3_channel = cv2.cvtColor(lbp_image_normalized, cv2.COLOR_GRAY2BGR)
    lbp_viz_3_channel = cv2.resize(lbp_viz_3_channel, FEATURE_IMG_SIZE) # Ensure display image is also consistent

    print(f"Extracting LBP features... (input shape: {lbp_input_for_model.shape})")
    # Return model input and visualization
    return lbp_input_for_model, Image.fromarray(lbp_viz_3_channel)


def extract_sift_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None # Return both input and viz
    
    img_resized = cv2.resize(img, FEATURE_IMG_SIZE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_resized, None)
    
    # --- Visualization of SIFT keypoints ---
    # Draw keypoints on a 3-channel version of the resized image
    img_display_with_kp = cv2.drawKeypoints(
        cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR), # Base image for drawing
        keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # SIFT model input (Bag of Visual Words histogram)
    if descriptors is None or len(keypoints) == 0:
        print("No SIFT descriptors found. Returning black image for input and visualization.")
        black_img_np = np.zeros(FEATURE_IMG_SIZE + (3,), dtype=np.uint8)
        # Ensure black_img_np is float32 for model input if that's its expectation
        return np.expand_dims(black_img_np, axis=0).astype(np.float32), Image.fromarray(black_img_np)

    global sift_kmeans_model # This line attempts to access a global variable
    if sift_kmeans_model is None: # Check the global variable directly
        print("Error: SIFT KMeans model not loaded. Returning black image for input and visualization.")
        black_img_np = np.zeros(FEATURE_IMG_SIZE + (3,), dtype=np.uint8)
        return np.expand_dims(black_img_np, axis=0).astype(np.float32), Image.fromarray(black_img_np)
    
    # Use the loaded global sift_kmeans_model
    visual_words = sift_kmeans_model.predict(descriptors)
    hist, _ = np.histogram(visual_words, bins=np.arange(sift_kmeans_model.n_clusters + 1))
    
    # The input to the SIFT model (Sift_model.h5) is likely the image with keypoints drawn
    # Ensure this image is float32 if the SIFT CNN model expects it.
    sift_input_for_model = np.expand_dims(img_display_with_kp, axis=0).astype(np.float32)
    
    print(f"SIFT features extracted with {len(keypoints)} keypoints (input shape: {sift_input_for_model.shape})")
    # Return model input and visualization
    return sift_input_for_model, Image.fromarray(img_display_with_kp)

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
                pred_index = 0 if predictions[0][0] < 0.5 else 1 # Assuming 0.5 threshold for binary
                class_channel = predictions 
                
        # Gradient of the output neuron with respect to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
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

def create_visualization_grid(original_image_pil, grad_cam_images, feature_visualizations=None, grid_size=(3, 4)): 
    """
    Creates a grid of visualizations including the original image, Grad-CAMs, and feature visualizations.
    """
    original_image_pil = original_image_pil.convert('RGB')
    
    img_width, img_height = original_image_pil.size
    cell_width = img_width
    cell_height = img_height

    all_images_to_display = {
        'Original Image': original_image_pil.resize((cell_width, cell_height))
    }

    # Add Grad-CAM images (unchanged logic)
    for name, img_pil in grad_cam_images.items():
        if isinstance(img_pil, Image.Image):
            all_images_to_display[f'Grad-CAM ({name})'] = img_pil.resize((cell_width, cell_height))
        else:
            placeholder_img = Image.new('RGB', (cell_width, cell_height), color = (200, 200, 200))
            draw = ImageDraw.Draw(placeholder_img)
            text = f"{name} (No CAM)"
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            # Use font.getbbox() instead of draw.textsize()
            bbox = font.getbbox(text) # Corrected to use 'text' and 'font'
            textwidth = bbox[2] - bbox[0]
            textheight = bbox[3] - bbox[1]
            x = (cell_width - textwidth) / 2
            y = (cell_height - textheight) / 2
            draw.text((x, y), text, fill=(0,0,0), font=font)
            all_images_to_display[f'Grad-CAM ({name})'] = placeholder_img

    # <--- ADDED: Logic to include Feature visualizations
    if feature_visualizations:
        for name, img_pil in feature_visualizations.items():
            if isinstance(img_pil, Image.Image):
                all_images_to_display[f'{name} Features'] = img_pil.resize((cell_width, cell_height))
            else:
                placeholder_img = Image.new('RGB', (cell_width, cell_height), color = (200, 200, 200))
                draw = ImageDraw.Draw(placeholder_img)
                text = f"{name} Features (No Viz)"
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                bbox = font.getbbox(text) # Corrected to use 'text' and 'font'
                textwidth = bbox[2] - bbox[0]
                textheight = bbox[3] - bbox[1]
                x = (cell_width - textwidth) / 2
                y = (cell_height - textheight) / 2
                draw.text((x, y), text, fill=(0,0,0), font=font)
                all_images_to_display[f'{name} Features'] = placeholder_img

    num_images = len(all_images_to_display)
    if num_images == 0:
        return Image.new('RGB', (cell_width, cell_height), color = 'white')

    if grid_size is None or grid_size[0] * grid_size[1] < num_images:
        cols = max(3, min(4, num_images))
        rows = (num_images + cols - 1) // cols
        grid_size = (rows, cols)

    grid_rows, grid_cols = grid_size
    
    final_grid_width = cell_width * grid_cols
    final_grid_height = cell_height * grid_rows + grid_rows * 30 # Add space for titles

    final_grid_image = Image.new('RGB', (final_grid_width, final_grid_height), color='white')
    draw = ImageDraw.Draw(final_grid_image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    x_offset = 0
    y_offset = 0
    img_counter = 0

    ordered_images = list(all_images_to_display.items())
    
    for title, img_pil in ordered_images:
        row = img_counter // grid_cols
        col = img_counter % grid_cols

        current_x = col * cell_width
        current_y = row * (cell_height + 30)

        final_grid_image.paste(img_pil, (current_x, current_y))

        bbox = font.getbbox(title) # Corrected to use 'title' and 'font'
        textwidth = bbox[2] - bbox[0]
        textheight = bbox[3] - bbox[1]
        text_x = current_x + (cell_width - textwidth) // 2
        text_y = current_y + cell_height + 5
        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)
        
        img_counter += 1

    return final_grid_image

def predict_image(img_path):
    """
    Runs prediction using the ensemble of image models, generates a comprehensive
    visualization grid, and calculates validation metrics if a true label is present.
    
    Returns:
        original_img_pil (PIL.Image): The original image.
        grid_img_pil (PIL.Image): The visualization grid image.
        result_text (str): The ensemble prediction text.
        ensemble_pred_prob (float): The ensemble prediction probability.
        predictions (dict): Dictionary of individual model probabilities.
        validation_metrics (dict): Dictionary of performance and trustability metrics.
    """
    if not models:
        messagebox.showerror("Error", "Models are not loaded.")
        return None, None, "Error: Models not loaded", None, {}, {} # Return empty dict for predictions and validation

    original_img_pil = Image.open(img_path).convert('RGB')
    original_img_cv = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR) # Keep for overlay

    predictions = {} # Stores probabilities for each model
    processed_data = {} # Store preprocessed data
    grad_cam_results = {} # Store Grad-CAM results for each applicable model
    feature_visualizations = {} # Store feature visualizations (HOG, LBP, SIFT)

    # --- Preprocess Image Data and Capture Visualizations ---
    processed_data['ResNet'] = preprocess_image_resnet(img_path)
    
    hog_input, hog_viz = extract_hog_features(img_path)
    processed_data['HOG'] = hog_input
    if hog_viz: feature_visualizations['HOG'] = hog_viz

    lbp_input, lbp_viz = extract_lbp_features(img_path)
    processed_data['LBP'] = lbp_input
    if lbp_viz: feature_visualizations['LBP'] = lbp_viz

    sift_input, sift_viz = extract_sift_features(img_path)
    processed_data['SIFT'] = sift_input
    if sift_viz: feature_visualizations['SIFT'] = sift_viz

    # Check if ResNet preprocessing failed (needed for Grad-CAM)
    if processed_data['ResNet'] is None:
        messagebox.showwarning("Warning", "Failed to preprocess image for ResNet/Grad-CAM.")
        # Allow continuing, but Grad-CAM might fail

    # --- Run Predictions ---
    for name, model in models.items():
        if model is None or name == 'sift_kmeans_model': # Skip KMeans model from direct prediction
            predictions[name] = None # Skip if model failed to load or is not a prediction model
            continue

        data = None
        # Select appropriate preprocessed data based on model name convention
        if 'ResNet' in name:
            data = processed_data['ResNet']
        elif 'Hog' in name:
            data = processed_data['HOG']
        elif 'LBP' in name:
            data = processed_data['LBP']
        elif 'Sift' in name: # This handles Sift models (e.g., Sift_model.h5)
            data = processed_data['SIFT']

        # Check if data is available for this model type
        if data is None:
            print(f"Skipping {name}: No suitable preprocessed data generated or model not loaded.")
            predictions[name] = None
            continue

        try:
            # Predict using the selected data
            pred = model.predict(data)[0] # Assuming batch size 1
            
            if len(pred) == 1:
                predictions[name] = float(pred[0]) # Assume single output is P(Cancer)
            elif len(pred) > 1:
                predictions[name] = float(pred[1]) # Assuming index 1 corresponds to the 'Cancer' class probability
            else:
                print(f"Warning: Unexpected prediction output shape for {name}: {pred.shape}")
                predictions[name] = None # Cannot interpret

            if predictions[name] is not None:
                print(f"{name} prediction: {predictions[name]:.4f}")
                
                # Generate Grad-CAM for ResNet models only
                if 'ResNet' in name:
                    try:
                        if hasattr(model, 'layers') and model.layers:
                            last_conv_layer_name = find_last_conv_layer(model)
                            if last_conv_layer_name:
                                try:
                                    model.get_layer(last_conv_layer_name) # Verify layer exists
                                    heatmap = get_grad_cam(model, data, last_conv_layer_name)
                                    if heatmap is not None:
                                        # Resize heatmap to match original image dimensions
                                        heatmap_resized = cv2.resize(heatmap, (original_img_cv.shape[1], original_img_cv.shape[0]))
                                        # Convert heatmap to RGB format
                                        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                                        # Overlay heatmap on original image
                                        alpha = 0.4   # Transparency factor
                                        overlay = cv2.addWeighted(original_img_cv, 1 - alpha, heatmap_colored, alpha, 0)
                                        
                                        # Store the result (convert from BGR to RGB PIL Image)
                                        grad_cam_results[name] = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                                        print(f"Generated Grad-CAM for {name}")
                                except Exception as e:
                                    print(f"Error generating Grad-CAM for {name} (layer issues or get_grad_cam): {e}")
                    except Exception as e:
                        print(f"Error during Grad-CAM setup for {name}: {e}")
            else:
                print(f"{name} prediction could not be interpreted.")

        except ValueError as ve:
            print(f"Error during prediction for {name} (Potential Shape Mismatch): {ve}")
            print(f"Model expected input shape: {model.input_shape}, Data shape provided: {data.shape}")
            predictions[name] = None
        except Exception as e:
            print(f"Error during prediction for {name}: {e}")
            predictions[name] = None

    # --- Ensemble Predictions ---
    valid_preds = [p for p in predictions.values() if p is not None]
    if not valid_preds:
        messagebox.showerror("Error", "No models could produce a valid prediction.")
        return original_img_pil, None, "Error: No valid predictions", None, predictions, {} # Return empty validation

    ensemble_pred_prob = np.mean(valid_preds)
    final_prediction = "Malignant" if ensemble_pred_prob > 0.5 else "Benign"
    result_text = f"{final_prediction} (Avg Prob: {ensemble_pred_prob:.3f})"
    print(f"\nEnsemble Probability (Avg): {ensemble_pred_prob:.4f}")
    print(f"Final Prediction: {final_prediction}")

    # --- Calculate Validation Metrics (New) ---
    validation_metrics = calculate_validation_metrics(predictions, img_path)
    print("\nValidation Metrics:")
    print(validation_metrics)

    # --- Create a Comprehensive Visualization Grid ---
    all_visualizations_for_grid = {}
    
    # Add original image
    all_visualizations_for_grid['Original Image'] = original_img_pil.resize(FEATURE_IMG_SIZE)

    # Add Feature Visualizations
    for name, viz_img in feature_visualizations.items():
        all_visualizations_for_grid[f'{name} Features'] = viz_img.resize(FEATURE_IMG_SIZE)

    # Add Grad-CAM Visualizations
    for model_name, cam_image_pil in grad_cam_results.items():
        if cam_image_pil is not None:
            all_visualizations_for_grid[f'{model_name} Grad-CAM'] = cam_image_pil.resize(FEATURE_IMG_SIZE)
        else:
            # Placeholder for failed Grad-CAM
            placeholder_img = Image.new('RGB', FEATURE_IMG_SIZE, color = (200, 200, 200))
            draw = ImageDraw.Draw(placeholder_img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            text = f"{model_name} (No CAM)"
            bbox = font.getbbox(text)
            textwidth = bbox[2] - bbox[0]
            textheight = bbox[3] - bbox[1]
            x = (FEATURE_IMG_SIZE[0] - textwidth) / 2
            y = (FEATURE_IMG_SIZE[1] - textheight) / 2
            draw.text((x, y), text, fill=(0,0,0), font=font)
            all_visualizations_for_grid[f'{model_name} Grad-CAM'] = placeholder_img

    num_images = len(all_visualizations_for_grid)
    if num_images == 0:
        grid_img_pil = Image.new('RGB', (FEATURE_IMG_SIZE[0], FEATURE_IMG_SIZE[1]), color='white')
        return original_img_pil, grid_img_pil, result_text, ensemble_pred_prob, predictions, validation_metrics

    # Determine grid dimensions dynamically
    grid_cols = max(1, min(4, num_images)) # Up to 4 columns, adjust as needed
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    final_grid_width = FEATURE_IMG_SIZE[0] * grid_cols
    final_grid_height = FEATURE_IMG_SIZE[1] * grid_rows + grid_rows * 30 # Add space for titles

    grid_img_pil = Image.new('RGB', (final_grid_width, final_grid_height), color='white')
    draw = ImageDraw.Draw(grid_img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    img_counter = 0
    # Sort images for consistent display order
    ordered_image_items = sorted(all_visualizations_for_grid.items())
    
    for title, img_pil in ordered_image_items:
        row = img_counter // grid_cols
        col = img_counter % grid_cols

        current_x = col * FEATURE_IMG_SIZE[0]
        current_y = row * (FEATURE_IMG_SIZE[1] + 30) # 30 for title space

        grid_img_pil.paste(img_pil, (current_x, current_y))

        bbox = font.getbbox(title) # Corrected to use 'title' and 'font'
        textwidth = bbox[2] - bbox[0]
        textheight = bbox[3] - bbox[1]
        text_x = current_x + (FEATURE_IMG_SIZE[0] - textwidth) // 2
        text_y = current_y + FEATURE_IMG_SIZE[1] + 5
        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)
        
        img_counter += 1
            
    return original_img_pil, grid_img_pil, result_text, ensemble_pred_prob, predictions, validation_metrics


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

        # Frame for images and metrics
        self.main_content_frame = tk.Frame(root)
        self.main_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Configure grid for main_content_frame: 2 columns, 1 row for images, 1 row for metrics
        self.main_content_frame.columnconfigure(0, weight=1) # Original Image
        self.main_content_frame.columnconfigure(1, weight=3) # Visualization Grid
        self.main_content_frame.rowconfigure(0, weight=3) # Images
        self.main_content_frame.rowconfigure(1, weight=1) # Metrics

        # Original image section
        tk.Label(self.main_content_frame, text="Original Image").grid(row=0, column=0, pady=2, sticky="n")
        self.input_canvas = tk.Canvas(self.main_content_frame, bg='lightgrey')
        self.input_canvas.grid(row=0, column=0, sticky="nsew", padx=5)

        # Comprehensive visualization grid section
        tk.Label(self.main_content_frame, text="All Model Predictions with Grad-CAM").grid(row=0, column=1, pady=2, sticky="n")
        self.output_canvas = tk.Canvas(self.main_content_frame, bg='lightgrey')
        self.output_canvas.grid(row=0, column=1, sticky="nsew", padx=5)

        # Add scrollbars for the visualization grid
        self.scrollbar_y = tk.Scrollbar(self.output_canvas, orient="vertical", command=self.output_canvas.yview)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_canvas.configure(yscrollcommand=self.scrollbar_y.set)

        self.scrollbar_x = tk.Scrollbar(self.output_canvas, orient="horizontal", command=self.output_canvas.xview)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.output_canvas.configure(xscrollcommand=self.scrollbar_x.set)

        # Metrics display area (new)
        self.metrics_frame = tk.LabelFrame(self.main_content_frame, text="Performance and Trustability Metrics")
        self.metrics_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=10)
        self.metrics_frame.grid_rowconfigure(0, weight=1)
        self.metrics_frame.grid_columnconfigure(0, weight=1)

        self.metrics_text = tk.Text(self.metrics_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 10))
        self.metrics_text.grid(row=0, column=0, sticky="nsew")

        self.metrics_scrollbar_y = tk.Scrollbar(self.metrics_frame, orient="vertical", command=self.metrics_text.yview)
        self.metrics_scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.metrics_text.config(yscrollcommand=self.metrics_scrollbar_y.set)

        # Initialize variables
        self.input_img_path = None
        self.gene_data_path = None
        self.input_img_display = None
        self.output_img_display = None
        self.input_img_pil = None
        self.output_img_pil = None

        # Bind resize event
        self.input_canvas.bind("<Configure>", lambda event: self.resize_image(event, self.input_canvas))
        self.output_canvas.bind("<Configure>", lambda event: self.resize_image(event, self.output_canvas))
        self.output_canvas.bind("<Configure>", self.on_output_canvas_configure) # For scrollable region

    def on_output_canvas_configure(self, event):
        # Update the scrollregion to encompass the entire content
        if self.output_img_pil:
            self.output_canvas.config(scrollregion=self.output_canvas.bbox(tk.ALL))


    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.input_img_path = path
            self.display_image(self.input_canvas, self.input_img_path)
            # Clear previous output and metrics
            self.output_canvas.delete("all")
            self.output_img_display = None
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.config(state=tk.DISABLED)
            self.result_label.config(text="Image loaded. Click Predict.")
            self.predict_button.config(state=tk.NORMAL) # Enable predict once image is loaded

    def load_gene_data(self):
        path = filedialog.askopenfilename(filetypes=[("Gene Data", "*.csv *.tsv *.txt")])
        if path:
            if not all(gene_pipeline.values()):
                 messagebox.showwarning("Warning", "Gene pipeline components not loaded. Cannot use gene data.")
                 self.gene_data_path = None
                 self.gene_file_label.config(text="Gene components missing.")
            else:
                self.gene_data_path = path
                self.gene_file_label.config(text=f"Gene Data: {os.path.basename(path)}")
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
            # Update scroll region for output canvas
            self.output_canvas.config(scrollregion=(0, 0, img_resized.width, img_resized.height))


    def run_prediction(self):
        """Runs prediction on the loaded image and displays results and validation metrics."""
        if not self.input_img_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        self.result_label.config(text="Predicting...")
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "Calculating metrics...\n")
        self.metrics_text.config(state=tk.DISABLED)
        self.root.update_idletasks() 

        try:
            # Updated to receive validation_metrics
            original_img_pil, grid_img_pil, result_text, ensemble_pred_prob, predictions_dict, validation_metrics = predict_image(self.input_img_path)

            if result_text: # Check if prediction was successful
                self.display_image(self.input_canvas, original_img_pil) # Use original_img_pil for input display
                self.display_image(self.output_canvas, grid_img_pil)    # Use grid_img_pil for output display
                self.result_label.config(text=f"Result: {result_text}")
                
                # Display individual model predictions and validation metrics
                self.metrics_text.config(state=tk.NORMAL)
                self.metrics_text.delete(1.0, tk.END)
                self.metrics_text.insert(tk.END, "--- Individual Model Predictions ---\n")
                for model_name, prob in predictions_dict.items():
                    if prob is not None:
                        self.metrics_text.insert(tk.END, f"{model_name}: Probability = {prob:.4f}\n")
                    else:
                        self.metrics_text.insert(tk.END, f"{model_name}: No prediction\n")

                self.metrics_text.insert(tk.END, "\n--- Ensemble Model ---\n")
                self.metrics_text.insert(tk.END, f"Ensemble Probability: {ensemble_pred_prob:.4f}\n")
                self.metrics_text.insert(tk.END, f"Final Ensemble Prediction: {'Malignant' if ensemble_pred_prob > 0.5 else 'Benign'}\n")

                self.metrics_text.insert(tk.END, "\n--- Validation Metrics ---\n")
                if validation_metrics["true_label_found"]:
                    self.metrics_text.insert(tk.END, f"True Label Found in Filename: {validation_metrics['true_label']} ({'Malignant' if validation_metrics['true_label'] == 1 else 'Benign'})\n\n")
                    
                    self.metrics_text.insert(tk.END, "Performance Metrics (vs. True Label):\n")
                    for model_name, perf_metrics in validation_metrics["performance"].items():
                        self.metrics_text.insert(tk.END, f"  {model_name}:\n")
                        for metric, value in perf_metrics.items():
                            self.metrics_text.insert(tk.END, f"    {metric.replace('_', ' ').title()}: {value:.4f}\n")
                    
                    self.metrics_text.insert(tk.END, "\nTrustability Metrics:\n")
                    for model_name, trust_metrics in validation_metrics["trustability"].items():
                        self.metrics_text.insert(tk.END, f"  {model_name}:\n")
                        self.metrics_text.insert(tk.END, f"    Predicted Class: {trust_metrics['predicted_class']} ({'Malignant' if trust_metrics['predicted_class'] == 1 else 'Benign'})\n")
                        self.metrics_text.insert(tk.END, f"    Is Correct: {trust_metrics['is_correct']}\n")
                        self.metrics_text.insert(tk.END, f"    Trust Score: {trust_metrics['trust_score']}\n")
                else:
                    self.metrics_text.insert(tk.END, "No true label found in filename. Cannot calculate performance and trustability metrics.\n")
                
                self.metrics_text.config(state=tk.DISABLED)

            else:
                self.result_label.config(text="Prediction failed. Check console for errors.")
                self.output_canvas.delete("all") # Clear output canvas on error
                self.metrics_text.config(state=tk.NORMAL)
                self.metrics_text.delete(1.0, tk.END)
                self.metrics_text.insert(tk.END, "Prediction failed. No metrics available.\n")
                self.metrics_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")
            self.result_label.config(text="Prediction failed.")
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, f"Error during prediction: {e}\n")
            self.metrics_text.config(state=tk.DISABLED)
            print(f"Prediction exception: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    if not load_all_models():
        print("Exiting due to model loading failure.")
    else:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
