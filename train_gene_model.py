import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.exceptions import NotFittedError
import joblib
import re
import os

# --- Configuration ---
DATA_FILE = 'e:\\bcrrp\\GSE1000_series_matrix.txt'
MODEL_SAVE_PATH = 'e:\\bcrrp\\models\\gene_expression_model.joblib'
SCALER_SAVE_PATH = 'e:\\bcrrp\\models\\gene_expression_scaler.joblib'
FEATURE_NAMES_PATH = 'e:\\bcrrp\\models\\gene_feature_names.joblib' # To store feature names

# --- Data Loading and Parsing ---
def load_geo_matrix(filepath):
    """
    Loads gene expression data from a GEO Series Matrix file.
    Extracts the expression matrix and sample metadata.
    """
    metadata = {}
    data_lines = []
    in_table = False
    header = None

    print(f"Loading data from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line == '!series_matrix_table_begin':
                    in_table = True
                    continue
                elif line == '!series_matrix_table_end':
                    in_table = False
                    break # Stop reading after table

                if in_table:
                    if header is None:
                        # Use regex to handle potential quotes around sample IDs
                        header = [re.sub(r'^"|"$', '', item) for item in line.split('\t')]
                    else:
                        data_lines.append(line.split('\t'))
                elif line.startswith('!Sample_'):
                    parts = line.split('\t')
                    key = parts[0]
                    values = [re.sub(r'^"|"$', '', item) for item in parts[1:]]
                    metadata[key] = values
                elif line.startswith('!Series_'):
                     # Store series metadata if needed, though less common for direct modeling
                     pass

        if not data_lines:
            raise ValueError("No data table found in the file.")

        # Create DataFrame
        df = pd.DataFrame(data_lines, columns=header)

        # Set gene IDs as index
        if 'ID_REF' in df.columns:
            df = df.set_index('ID_REF')
        else:
            raise ValueError("ID_REF column not found in the table header.")

        # Convert data to numeric, coercing errors
        df = df.apply(pd.to_numeric, errors='coerce')

        # Drop rows/genes with any NaN values (resulting from coercion errors or missing data)
        df = df.dropna(axis=0)

        # Transpose: Samples as rows, Genes as columns
        df = df.T

        print(f"Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} genes/features.")
        return df, metadata

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error loading or parsing file: {e}")
        return None, None

# --- Preprocessing ---
def preprocess_data(df):
    """Scales the gene expression data."""
    print("Scaling data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    print("Data scaling complete.")
    return scaled_df, scaler

# --- Feature Engineering / Label Creation (EXAMPLE ONLY) ---
def create_labels_example(df, metadata):
    """
    *** EXAMPLE FUNCTION - NEEDS REAL LABELS ***
    Creates dummy labels based on metadata.
    Replace this with logic based on your actual target variable.
    This example tries to classify based on time point (6 vs 32 hours).
    """
    print("WARNING: Creating EXAMPLE labels based on time points (6h vs 32h).")
    print("         This is NOT cancer vs. normal prediction.")

    sample_titles = metadata.get('!Sample_title')
    sample_ids = df.index # Sample IDs are now the index after transpose

    if sample_titles is None or len(sample_titles) != len(sample_ids):
         print("Error: Metadata '!Sample_title' missing or doesn't match data samples.")
         print("Cannot create example labels.")
         return None

    # Create a mapping from sample ID to title (assuming order matches initially)
    # The header line in the file gives the sample IDs corresponding to the columns
    # After transposing, these IDs are the index.
    # We need to ensure the metadata order corresponds correctly.
    # Let's assume the order in !Sample_title corresponds to the column order in the original file.
    title_map = {sample_id: title for sample_id, title in zip(metadata.get('!Sample_geo_accession', []), sample_titles)}

    labels = []
    for sample_id in sample_ids:
        title = title_map.get(sample_id)
        if title is None:
            print(f"Warning: No title found for sample {sample_id}. Skipping.")
            labels.append(np.nan) # Mark for potential removal
        elif '6 hours' in title:
            labels.append(0) # Assign 0 for 6 hours
        elif '32 hours' in title:
            labels.append(1) # Assign 1 for 32 hours
        else:
            print(f"Warning: Could not determine time point for sample {sample_id} ('{title}'). Skipping.")
            labels.append(np.nan) # Mark for potential removal

    labels = pd.Series(labels, index=sample_ids)

    # Remove samples where labels couldn't be determined
    valid_indices = labels.dropna().index
    if len(valid_indices) < len(df):
        print(f"Removing {len(df) - len(valid_indices)} samples due to missing labels.")
        df = df.loc[valid_indices]
        labels = labels.loc[valid_indices]

    if len(labels.unique()) < 2:
        print("Error: Could not create at least two distinct classes for training.")
        return None

    print("Example labels created:")
    print(labels.value_counts())
    return labels.astype(int)


# --- Model Training ---
def train_model(X, y):
    """Trains a Logistic Regression model."""
    if X.empty or y.empty:
        print("Error: Input data or labels are empty. Cannot train model.")
        return None

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter for convergence
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

# --- Saving ---
def save_pipeline(model, scaler, feature_names, model_path, scaler_path, features_path):
    """Saves the trained model, scaler, and feature names."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)

        print(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        print(f"Saving scaler to {scaler_path}")
        joblib.dump(scaler, scaler_path)
        print(f"Saving feature names to {features_path}")
        joblib.dump(feature_names, features_path) # Save the column names
        print("Pipeline components saved successfully.")
    except Exception as e:
        print(f"Error saving pipeline components: {e}")

# --- Prediction Function ---
def predict_new_sample(new_data_row, model_path, scaler_path, features_path):
    """
    Loads the pipeline and predicts on new data.
    'new_data_row' should be a pandas Series or DataFrame row
    with gene IDs matching the training data features.
    """
    try:
        print("\n--- New Prediction ---")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        expected_features = joblib.load(features_path)
        print("Loaded model, scaler, and feature names.")

        # Ensure new_data_row is a DataFrame with the correct features
        if isinstance(new_data_row, pd.Series):
            new_data_df = pd.DataFrame(new_data_row).T
        elif isinstance(new_data_row, pd.DataFrame):
            new_data_df = new_data_row
        else:
             raise ValueError("Input 'new_data_row' must be a pandas Series or DataFrame.")

        # Align columns - crucial step!
        # Add missing columns with 0 (or mean/median if appropriate, but 0 is safer if unsure)
        # Remove extra columns
        new_data_aligned = new_data_df.reindex(columns=expected_features, fill_value=0)

        if new_data_aligned.shape[1] != len(expected_features):
             raise ValueError(f"Input data has {new_data_aligned.shape[1]} features, but model expects {len(expected_features)}.")

        # Check for NaNs introduced by alignment or original data
        if new_data_aligned.isnull().values.any():
            print("Warning: Input data contains NaN values. Filling with 0.")
            new_data_aligned = new_data_aligned.fillna(0)

        # Apply the *same* scaling
        scaled_new_data = scaler.transform(new_data_aligned) # Use transform, not fit_transform!

        # Predict
        prediction = model.predict(scaled_new_data)
        prediction_proba = model.predict_proba(scaled_new_data)

        print(f"Input data shape after alignment: {new_data_aligned.shape}")
        # Interpret prediction based on the EXAMPLE labels (0: 6h, 1: 32h)
        # ** MODIFY THIS INTERPRETATION BASED ON YOUR ACTUAL LABELS **
        predicted_label = "Condition 1 (e.g., 32 hours)" if prediction[0] == 1 else "Condition 0 (e.g., 6 hours)"
        print(f"Predicted Class: {prediction[0]} ({predicted_label})")
        print(f"Prediction Probabilities: {prediction_proba[0]}")
        return prediction[0], prediction_proba[0]

    except FileNotFoundError:
        print("Error: Model, scaler, or feature names file not found. Train the model first.")
        return None, None
    except NotFittedError:
         print("Error: Scaler was not fitted. Train the model first.")
         return None, None
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    expression_df, metadata_dict = load_geo_matrix(DATA_FILE)

    if expression_df is not None:
        # 2. Create Labels (Using the EXAMPLE function)
        # !!! Replace create_labels_example with your actual label creation logic !!!
        labels = create_labels_example(expression_df, metadata_dict)

        if labels is not None:
             # Ensure expression_df matches labels after potential removals
             expression_df = expression_df.loc[labels.index]

             # Store feature names before scaling (column names of expression_df)
             feature_names = expression_df.columns.tolist()

             # 3. Preprocess Data (Scaling)
             scaled_expression_df, data_scaler = preprocess_data(expression_df)

             # 4. Train Model
             trained_model = train_model(scaled_expression_df, labels)

             if trained_model:
                 # 5. Save Pipeline
                 save_pipeline(trained_model, data_scaler, feature_names,
                               MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURE_NAMES_PATH)

                 # 6. Example Prediction (using the first sample from the original data)
                 if not expression_df.empty:
                     print("\n--- Running example prediction on the first data sample ---")
                     # Use the *original* unscaled data for the prediction function input
                     first_sample_unscaled = expression_df.iloc[[0]]
                     predict_new_sample(first_sample_unscaled, MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURE_NAMES_PATH)
                 else:
                     print("\nSkipping example prediction as data is empty.")
             else:
                 print("\nModel training failed. Pipeline not saved.")
        else:
             print("\nLabel creation failed. Cannot proceed with training.")
    else:
        print("\nData loading failed. Exiting.")