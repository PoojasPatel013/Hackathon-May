# Create a comprehensive disaster risk prediction module
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDisasterRiskModel(nn.Module):
    """
    Neural network model for disaster risk prediction
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

class DisasterRiskPredictor:
    """
    Comprehensive disaster risk prediction predictor
    Handles model loading, configuration, and risk predictions
    """
    def __init__(self, config_path=None, model_path=None):
        """
        Initialize predictor with workflow configuration and model path
        
        Args:
            config_path (str, optional): Path to workflow configuration
            model_path (str, optional): Path to saved model pickle file
        """
        # Define possible config paths
        if config_path is None:
            possible_config_paths = [
                os.path.join(os.path.dirname(__file__), 'config', 'workflow_config.yaml'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'workflow_config.yaml'),
                'workflow_config.yaml'
            ]
            
            # Find the first existing config path
            config_path = next((path for path in possible_config_paths if os.path.exists(path)), None)
            
            if config_path is None:
                # Use default configuration if no file found
                config_path = self._create_default_config()
        
        # Define possible model paths
        if model_path is None:
            # Look for models in multiple directories
            model_dirs = [
                os.path.join(os.path.dirname(__file__), 'models'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
                os.path.join(os.path.dirname(__file__), '..', 'model'),
                os.path.join(os.path.dirname(__file__), 'model')
            ]
            
            # Find all pickle files in model directories
            model_files = []
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    model_files.extend([
                        os.path.join(model_dir, f) 
                        for f in os.listdir(model_dir) 
                        if f.endswith('.pkl')
                    ])
            
            # If multiple models found, log and use the first one
            if model_files:
                if len(model_files) > 1:
                    logger.warning(f"Multiple model files found: {model_files}. Using the first one.")
                model_path = model_files[0]
            else:
                raise FileNotFoundError("No model files found in expected directories")
        
        # Load workflow configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            self.config = {}
        
        # Load saved components
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Dynamically import the model class
            import importlib
            
            # Try different import strategies
            import_strategies = [
                'predict',  # Current module
                'frontend.predict',  # Project-level module
                'streamlit_app.predict'  # Streamlit app module
            ]
            
            model_class = None
            for module_name in import_strategies:
                try:
                    module = importlib.import_module(module_name)
                    model_class = getattr(module, saved_data.get('model_class_name', 'SimpleDisasterRiskModel'), None)
                    if model_class:
                        break
                except ImportError:
                    continue
            
            if model_class is None:
                raise ImportError(f"Could not find model class {saved_data.get('model_class_name', 'SimpleDisasterRiskModel')}")
            
            # Recreate model
            self.model = model_class(
                saved_data['input_dim'], 
                saved_data['num_classes']
            )
            self.model.load_state_dict(saved_data['model_state'])
            self.model.eval()
            
            # Save other components
            self.scaler = saved_data['scaler']
            self.label_encoder = saved_data['label_encoder']
            self.feature_names = saved_data['feature_names']
            
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def _create_default_config(self):
        """
        Create a default workflow configuration if no config file is found
        
        Returns:
            str: Path to the created default configuration file
        """
        default_config = {
            'workflow': {
                'name': 'Disaster Risk Prediction',
                'version': '0.1.0'
            },
            'data_collection': {
                'sources': {
                    'search_directories': ['data']
                },
                'preprocessing': {
                    'drop_missing': True
                }
            },
            'predictions': {
                'risk_categories': [
                    'Very Low Risk', 
                    'Low Risk', 
                    'Moderate Risk', 
                    'High Risk', 
                    'Extreme Risk'
                ]
            }
        }
        
        # Ensure config directory exists
        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Create default config file
        config_path = os.path.join(config_dir, 'workflow_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        return config_path
    
    def predict(self, input_features):
        """
        Make a prediction based on input features
        
        Args:
            input_features (list): List of input feature values
        
        Returns:
            dict: Prediction results with risk level and probabilities
        """
        # Ensure input matches feature names
        if len(input_features) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(input_features)}")
        
        # Scale input features
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = self.scaler.transform(input_array)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_scaled)
        
        # Make prediction
        with torch.no_grad():
            probabilities = self.model(input_tensor).numpy()[0]
        
        # Get risk categories from config
        risk_categories = self.config['predictions'].get('risk_categories', 
            ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']
        )
        
        # Determine risk level
        risk_index = np.argmax(probabilities)
        risk_level = {
            'label': risk_categories[risk_index],
            'description': f"{risk_categories[risk_index]} Risk Level",
            'probability': probabilities[risk_index]
        }
        
        return {
            'risk_level': risk_level,
            'probabilities': probabilities.tolist()
        }
    
    def predict_global_risks(self):
        """
        Generate global disaster risk predictions
        
        Returns:
            list: Risk predictions for different regions
        """
        # Placeholder for global risk prediction logic
        # In a real-world scenario, this would use actual global data
        global_data = np.random.randn(10, len(self.feature_names))
        global_predictions = []
        
        for features in global_data:
            prediction = self.predict(features)
            global_predictions.append(prediction['risk_level']['label'])
        
        return global_predictions

def main():
    """
    Main function to demonstrate model loading and prediction
    """
    try:
        # Initialize predictor
        predictor = DisasterRiskPredictor()
        
        # Generate some mock features for prediction
        mock_features = np.random.randn(len(predictor.feature_names)).tolist()
        
        # Make a prediction
        prediction = predictor.predict(mock_features)
        print("Prediction Result:")
        print(f"Risk Level: {prediction['risk_level']['label']}")
        print(f"Probabilities: {prediction['probabilities']}")
        
        # Generate global risks
        global_risks = predictor.predict_global_risks()
        print("\nGlobal Risk Predictions:")
        print(global_risks)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()