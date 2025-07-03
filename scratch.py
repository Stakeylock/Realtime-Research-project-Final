
Python


# Example snippet from inference_app.py (conceptual for gene data)
import pandas as pd
import joblib
import os

# Load persisted scaler and feature names
GENE_SCALER_PATH = 'models/gene_expression_scaler.joblib'
GENE_FEATURE_NAMES_PATH = 'models/gene_feature_names.joblib'
GENE_MODEL_PATH = 'models/gene_expression_model.joblib' # Or gene_expression_model_tuned.joblib

def predict_gene_expression_data(file_path):
    if file_path.endswith('.csv'):
        gene_df = pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        gene_df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError("Unsupported file format. Please upload CSV or TSV.")

    scaler = joblib.load(GENE_SCALER_PATH)
    feature_names = joblib.load(GENE_FEATURE_NAMES_PATH)
    model = joblib.load(GENE_MODEL_PATH)
    gene_df_aligned = gene_df[feature_names]

    scaled_gene_data = scaler.transform(gene_df_aligned)

    # Predict class and probability
    predicted_class = model.predict(scaled_gene_data)
    prediction_probability = model.predict_proba(scaled_gene_data).tolist()

    return predicted_class, prediction_probability
