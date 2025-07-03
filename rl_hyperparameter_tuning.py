import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# Replace gym imports with gymnasium
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import glob
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import time
import logging
import warnings # Added import

# Suppress the specific UserWarning from sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("rl_tuning.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Constants
RL_MODELS_DIR = "e:\\bcrrp\\RL_Models"
MODELS_DIR = "e:\\bcrrp\\models"
OUTPUT_FILES_DIR = "e:\\bcrrp\\Output_Files"
GENE_DATA_FILE = "e:\\bcrrp\\GSE1000_series_matrix.txt"

# Create RL_Models directory if it doesn't exist
os.makedirs(RL_MODELS_DIR, exist_ok=True)

# Helper function to load gene expression data
def load_gene_data():
    logger.info("Loading gene expression data...")
    try:
        # Load gene expression data
        # This is a simplified approach - you may need to adjust based on your data format
        with open(GENE_DATA_FILE, 'r') as f:
            lines = f.readlines()
        
        # Find the start of the data section
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('!series_matrix_table_begin'):
                data_start = i + 1
                break
        
        # Extract header and data
        header = lines[data_start].strip().split('\t')
        data = []
        for line in lines[data_start+1:]:
            if line.startswith('!series_matrix_table_end'):
                break
            data.append(line.strip().split('\t'))
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=header)
        
        # Assuming the first column contains gene IDs and the rest are samples
        gene_ids = df.iloc[:, 0]
        expression_data = df.iloc[:, 1:].astype(float)
        
        # Create labels (assuming binary classification - modify as needed)
        # This is a placeholder - you'll need to adjust based on your actual labels
        labels = np.random.randint(0, 2, size=expression_data.shape[1])
        
        return expression_data.T, labels, gene_ids
    except Exception as e:
        logger.error(f"Error loading gene data: {e}")
        raise

# Helper function to load image data
def load_image_data(model_type):
    logger.info(f"Loading image data for {model_type}...")
    try:
        # Determine the appropriate directory based on model type
        if 'Hog' in model_type:
            if 'AHE' in model_type:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'HogAHE_Images')
            elif 'N' in model_type:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'HogNeg_Images')
            else:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'Hog_Images')
        elif 'LBP' in model_type:
            if 'AHE' in model_type:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'LBPAHE_images')
            elif 'N' in model_type:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'LBPNeg_images')
            else:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'LBP_images')
        elif 'Sift' in model_type:
            if 'AHE' in model_type:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'SiftAHE_Images')
            elif 'N' in model_type:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'SiftNeg_Images')
            else:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'SIFT_Images')
        elif 'resnet' in model_type:
            # For ResNet, we'll use the original images
            if 'AHE' in model_type:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'AHistogram_Images')
            elif 'N' in model_type:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'Negative_Images')
            else:
                img_dir = os.path.join(OUTPUT_FILES_DIR, 'merged_images')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create data generators
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        
        # Load training data
        train_generator = datagen.flow_from_directory(
            img_dir,
            target_size=(224, 224),  # Adjust based on your model's input size
            batch_size=32,
            class_mode='binary',
            subset='training'
        )
        
        # Load validation data
        val_generator = datagen.flow_from_directory(
            img_dir,
            target_size=(224, 224),  # Adjust based on your model's input size
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, val_generator
    except Exception as e:
        logger.error(f"Error loading image data for {model_type}: {e}")
        raise

# Environment for gene expression model tuning
class GeneModelTuningEnv(gym.Env):
    def __init__(self, model_path, scaler_path, feature_names_path, X, y):
        super().__init__()
        
        # Load the model, scaler, and feature names
        self.base_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_names_path)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Define action and observation spaces
        # Actions: C, max_iter, tol
        self.action_space = spaces.Box(
            low=np.array([0.1, 100, 1e-5]),
            high=np.array([10.0, 1000, 1e-3]),
            dtype=np.float32
        )
        
        # Observation space: current performance metrics
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 50
        self.best_score = 0
        self.best_params = None
        self.current_model = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Initialize RNG state
        self.current_step = 0
        # Return initial observation (current model performance)
        self.current_model = self.base_model
        y_pred = self.current_model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        return np.array([accuracy, f1], dtype=np.float32), {}  # Return observation and empty info dict
    
    def step(self, action):
        self.current_step += 1
        
        # Extract hyperparameters from action
        C, max_iter, tol = action
        

        # Create a new model with the selected hyperparameters
        model_class = type(self.base_model)
        if not hasattr(model_class, 'fit'): 
            logger.error(f"Base model type {model_class} might not be a scikit-learn classifier.")
        
        model = model_class( 
            C=float(C),
            max_iter=int(max_iter), 
            tol=float(tol),
            random_state=42 
        )
        
        # Train the model
        try:
            model.fit(self.X_train_scaled, self.y_train)
        except Exception as e:
            logger.error(f"Error fitting gene model with params C={C}, max_iter={max_iter}, tol={tol}: {e}")
            obs = np.array([0, 0], dtype=np.float32) 
            reward = -100 
            info = {'error': str(e)}
            return obs, reward, True, False, info # terminated=True, truncated=False
        
        # Evaluate the model
        y_pred = model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Calculate reward (weighted combination of metrics)
        reward = 0.7 * accuracy + 0.3 * f1
        
        # Update best model if this one is better
        if reward > self.best_score:
            self.best_score = reward
            self.best_params = {'C': C, 'max_iter': int(max_iter), 'tol': tol}
            self.current_model = model
        
        # Check if episode is done
        terminated = False # Set to True if a task-specific terminal condition is met
        truncated = self.current_step >= self.max_steps # Set to True if max_steps reached
        
        obs = np.array([accuracy, f1], dtype=np.float32)
        info = {
            'accuracy': accuracy,
            'f1_score': f1,
            'best_score': self.best_score,
            'best_params': self.best_params
        }
        
        return obs, reward, terminated, truncated, info
    
    def save_best_model(self, save_path):
        if self.current_model is not None:
            joblib.dump(self.current_model, save_path)
            logger.info(f"Saved best model to {save_path} with params {self.best_params}")
            return True
        return False

# Environment for image model tuning
class ImageModelTuningEnv(gym.Env):
    def __init__(self, model_path, train_gen, val_gen):
        super().__init__()
        
        # Load the base model
        self.base_model = load_model(model_path)
        self.train_gen = train_gen
        self.val_gen = val_gen
        
        # Define action and observation spaces
        # Actions: learning_rate, batch_size, dropout_rate
        self.action_space = spaces.Box(
            low=np.array([1e-5, 8, 0.1]),
            high=np.array([1e-2, 64, 0.5]),
            dtype=np.float32
        )
        
        # Observation space: current performance metrics (val_accuracy, val_loss)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, 10]),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 20
        self.best_score = 0
        self.best_params = None
        self.current_model = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Initialize RNG state
        self.current_step = 0
        # Return initial observation (current model performance)
        self.current_model = self.base_model
        
        # Evaluate the base model
        # Ensure evaluation is done correctly for Keras models
        if hasattr(self.current_model, 'evaluate'):
            evaluation = self.current_model.evaluate(self.val_gen, verbose=0) # Add verbose=0
            val_loss, val_accuracy = evaluation[0], evaluation[1]
        else:
            # Handle non-Keras models if necessary, or assume Keras for now
            logger.warning("Base model in ImageModelTuningEnv does not have 'evaluate' method. Using dummy values.")
            val_loss, val_accuracy = 10.0, 0.0 # Dummy values
            
        return np.array([val_accuracy, val_loss], dtype=np.float32), {}
    
    def step(self, action):
        self.current_step += 1
        
        # Extract hyperparameters from action
        learning_rate, batch_size, dropout_rate = action
        batch_size = int(batch_size) 
        
        if not isinstance(self.base_model, tf.keras.Model):
            logger.error("Base model is not a Keras model. Cannot clone.")
            obs = np.array([0, 10.0], dtype=np.float32) 
            reward = -100
            info = {'error': 'Base model not a Keras model'}
            return obs, reward, True, False, info # terminated=True, truncated=False

        model = tf.keras.models.clone_model(self.base_model)
        
        # Apply dropout to dense layers
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Dense) and i < len(model.layers) - 1:  # Skip output layer
                config = layer.get_config()
                if 'dropout' not in config:
                    # Add dropout after this layer
                    x = layer.output
                    x = tf.keras.layers.Dropout(dropout_rate)(x)
                    # Update the layer's output
                    layer._outbound_nodes = []
                    layer._outbound_nodes.append(x._keras_history.layer)
        
        # Compile the model with the new learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate)) # Ensure learning_rate is float
        model.compile(
            optimizer=optimizer, # Use the defined optimizer
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Update batch size in generators
        self.train_gen.batch_size = batch_size
        
        # Train for a few epochs
        history = model.fit(
            self.train_gen,
            epochs=3,
            validation_data=self.val_gen,
            verbose=0
        )
        
        # Get the final validation metrics
        val_accuracy = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        
        # Calculate reward (prioritize accuracy but penalize high loss)
        reward = val_accuracy - 0.1 * val_loss
        
        # Update best model if this one is better
        if reward > self.best_score:
            self.best_score = reward
            self.best_params = {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate
            }
            self.current_model = model
        
        # Check if episode is done
        max_steps_reached = self.current_step >= self.max_steps
        
        terminated = False # Assuming termination only on error or if explicitly set
        truncated = max_steps_reached # Truncated if max_steps reached
        
        obs = np.array([val_accuracy, val_loss], dtype=np.float32)
        info = {
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'best_score': self.best_score,
            'best_params': self.best_params
        }
        
        return obs, reward, terminated, truncated, info
    
    def save_best_model(self, save_path):
        if self.current_model is not None:
            self.current_model.save(save_path)
            logger.info(f"Saved best model to {save_path} with params {self.best_params}")
            return True
        return False

# Function to tune gene expression model
def tune_gene_model():
    logger.info("Starting gene expression model tuning...")
    
    # Load data
    try:
        X, y, gene_ids = load_gene_data()
        if X is None or y is None:
            logger.error("Failed to load gene data. Skipping gene model tuning.")
            return
    except Exception as e:
        logger.error(f"Error loading gene data: {e}. Skipping gene model tuning.")
        return

    # Paths to model files
    model_path = os.path.join(MODELS_DIR, "gene_expression_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "gene_expression_scaler.joblib")
    feature_names_path = os.path.join(MODELS_DIR, "gene_feature_names.joblib")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_names_path]):
        logger.error("One or more gene model files (model, scaler, features) not found. Skipping tuning.")
        return
        
    # Create the environment
    env = GeneModelTuningEnv(model_path, scaler_path, feature_names_path, X, y)
    env = DummyVecEnv([lambda: env])
    
    # Create the RL agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, device='cpu') # Explicitly use CPU
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(RL_MODELS_DIR, "gene_model_best_rl"),
        log_path=os.path.join(RL_MODELS_DIR, "gene_model_logs"),
        eval_freq=500,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    model.learn(total_timesteps=10000, callback=eval_callback)
    
    # Save the final model
    model.save(os.path.join(RL_MODELS_DIR, "gene_expression_rl_agent"))
    
    # Save the best tuned gene model
    env.envs[0].save_best_model(os.path.join(RL_MODELS_DIR, "gene_expression_model_tuned.joblib"))
    
    logger.info("Gene expression model tuning completed.")

# Function to tune image models
def tune_image_model(model_name):
    logger.info(f"Starting {model_name} model tuning...")
    
    # Load data
    try:
        train_gen, val_gen = load_image_data(model_name)
        if train_gen is None or val_gen is None:
            logger.error(f"Failed to load image data for {model_name}. Skipping tuning.")
            return
    except Exception as e:
        logger.error(f"Error loading image data for {model_name}: {e}. Skipping tuning.")
        return
    
    # Path to model file
    model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found for {model_name}. Skipping tuning.")
        return
        
    # Create the environment
    env = ImageModelTuningEnv(model_path, train_gen, val_gen)
    env = DummyVecEnv([lambda: env])
    
    # Create the RL agent
    # In tune_gene_model function, modify the PPO creation:
    # Create the RL agent with CPU device
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, device='cpu')
    
    # In ImageModelTuningEnv class, modify the data generator setup:
    # Update the class_mode in load_image_data function
    train_generator = datagen.flow_from_directory(
        img_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',  # Change from 'binary' to 'categorical'
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        img_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',  # Change from 'binary' to 'categorical'
        subset='validation'
    )
    
    # In ImageModelTuningEnv class, update model compilation:
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',  # Change from 'binary_crossentropy'
        metrics=['accuracy']
    )
    
    # In tune_image_model function, modify PPO creation:
    # Create the RL agent with CPU device
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, device='cpu')
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(RL_MODELS_DIR, f"{model_name}_best_rl"),
        log_path=os.path.join(RL_MODELS_DIR, f"{model_name}_logs"),
        eval_freq=500,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    model.learn(total_timesteps=5000, callback=eval_callback)
    
    # Save the final agent
    model.save(os.path.join(RL_MODELS_DIR, f"{model_name}_rl_agent"))
    
    # Save the best tuned model
    env.envs[0].save_best_model(os.path.join(RL_MODELS_DIR, f"{model_name}_tuned.h5"))
    
    logger.info(f"{model_name} model tuning completed.")

# Main function
def main():
    logger.info("Starting hyperparameter tuning with reinforcement learning...")
    
    # Get list of all models
    if not os.path.exists(MODELS_DIR):
        logger.error(f"Models directory {MODELS_DIR} not found. Aborting.")
        return
    model_files = os.listdir(MODELS_DIR)
    
    # Tune gene expression model
    if "gene_expression_model.joblib" in model_files:
        try:
            tune_gene_model()
        except Exception as e:
            logger.error(f"Unhandled error tuning gene expression model: {e}", exc_info=True)
    else:
        logger.info("Gene expression model not found in models directory. Skipping its tuning.")
        
    # Tune image models
    image_model_names = [
        "Hog_model", "HogAHE_model", "HogN_model",
        "LBP_model", "LBPAHE_model", "LBPN_model",
        "Sift_model", "SiftAHE_model", "SiftN_model", # Assuming Sift models exist
        "resnet_model", "resnetAHE_model", "resnetN_model"
    ]
    
    for model_name in image_model_names:
        if f"{model_name}.h5" in model_files:
            try:
                tune_image_model(model_name)
            except Exception as e:
                logger.error(f"Unhandled error tuning {model_name}: {e}", exc_info=True)
        else:
            logger.info(f"Image model {model_name}.h5 not found. Skipping its tuning.")

if __name__ == "__main__":
    main()