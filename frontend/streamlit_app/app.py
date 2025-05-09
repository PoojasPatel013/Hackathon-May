import os
import sys
import streamlit as st
import yaml
import logging
import pickle
import torch
import torch.nn as nn
import numpy as np
from logging.config import dictConfig

# Add the parent directory to Python path to make imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Print paths for debugging
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Python path: {sys.path}")

# Define the model class directly in app.py to match the pickle file
class DisasterRiskNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        # Feature extractor layers
        self.feature_extractor_0 = nn.Linear(input_dim, 64)
        self.feature_extractor_bn_0 = nn.BatchNorm1d(64)
        self.feature_extractor_relu_0 = nn.ReLU()
        self.feature_extractor_dropout_0 = nn.Dropout(0.3)
        
        self.feature_extractor_1 = nn.Linear(64, 64)
        self.feature_extractor_bn_1 = nn.BatchNorm1d(64)
        self.feature_extractor_relu_1 = nn.ReLU()
        self.feature_extractor_dropout_1 = nn.Dropout(0.3)
        
        # Last layer of feature extractor (without BatchNorm or Dropout)
        self.feature_extractor_2 = nn.Linear(64, 32)
        
        # Classifier layers
        self.classifier_0 = nn.Linear(32, 16)
        self.classifier_relu_0 = nn.ReLU()
        self.classifier_1 = nn.Linear(16, num_classes)
        self.classifier_softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Feature extractor
        x = self.feature_extractor_0(x)
        x = self.feature_extractor_bn_0(x)
        x = self.feature_extractor_relu_0(x)
        x = self.feature_extractor_dropout_0(x)
        
        x = self.feature_extractor_1(x)
        x = self.feature_extractor_bn_1(x)
        x = self.feature_extractor_relu_1(x)
        x = self.feature_extractor_dropout_1(x)
        
        x = self.feature_extractor_2(x)
        
        # Classifier
        x = self.classifier_0(x)
        x = self.classifier_relu_0(x)
        x = self.classifier_1(x)
        x = self.classifier_softmax(x)
        
        return x

# Create a custom unpickler that will use our DisasterRiskNetwork class
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # If the class is DisasterRiskNetwork, return our version
        if module == "__main__" and name == "DisasterRiskNetwork":
            return DisasterRiskNetwork
        # For everything else, use the default behavior
        return super().find_class(module, name)

# Import components using direct imports
from streamlit_app.components.risk_assessment import disaster_risk_prediction_page
from streamlit_app.components.map_visualization import generate_risk_map

class DisasterRiskPredictor:
    """
    Disaster Risk Prediction using pre-trained pickle model
    """
    def __init__(self, model_path: str = None):
        """
        Initialize predictor with model from pickle file
        
        Args:
            model_path (str, optional): Path to saved model pickle file
        """
        # Find model path if not provided
        if model_path is None:
            model_path = self._find_model_path()
        
        # Load model and components
        self._load_model(model_path)
        
        # Default configuration
        self.config = {
            'predictions': {
                'risk_categories': ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']
            }
        }
    
    def _find_model_path(self) -> str:
        """
        Find the first existing model pickle file
        
        Returns:
            str: Path to the model file
        """
        # Current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Possible model directories
        model_dirs = [
            os.path.join(current_dir, 'model'),
            os.path.join(os.path.dirname(current_dir), 'model'),
            os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'model')
        ]
        
        # Search for pickle files
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                model_files = [
                    os.path.join(model_dir, f) 
                    for f in os.listdir(model_dir) 
                    if f.endswith('.pkl')
                ]
                if model_files:
                    return model_files[0]
        
        raise FileNotFoundError("No model pickle file found")
    
    def _load_model(self, model_path: str):
        """
        Load model, scaler, and label encoder from pickle file
        
        Args:
            model_path (str): Path to model pickle file
        """
        try:
            # Print full path for debugging
            print(f"Attempting to load model from: {os.path.abspath(model_path)}")
            
            # Verify file exists and is readable
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file does not exist: {model_path}")
            
            # Load pickle file with custom unpickler
            with open(model_path, 'rb') as f:
                saved_data = CustomUnpickler(f).load()
            
            # Extract components
            if 'model_state' in saved_data:
                # Create model instance
                self.model = DisasterRiskNetwork(
                    input_dim=saved_data['input_dim'], 
                    num_classes=saved_data['num_classes']
                )
                # Load model state
                self.model.load_state_dict(saved_data['model_state'], strict=False)
                self.model.eval()
            else:
                # If the model is directly saved
                self.model = saved_data.get('model', None)
            
            # Extract other components
            self.scaler = saved_data.get('scaler', None)
            self.label_encoder = saved_data.get('label_encoder', None)
            self.feature_names = saved_data.get('feature_names', [
                'Magnitude', 'Depth', 'Wind Speed', 'Tsunami Intensity', 'Significance'
            ])
            
            print("Model loaded successfully with all components")
        
        except Exception as e:
            print(f"Comprehensive error loading model from {model_path}: {e}")
            raise
    
    def predict(self, input_features):
        """
        Make a prediction based on input features
        
        Args:
            input_features: Input feature values
        
        Returns:
            Dict: Prediction results
        """
        try:
            # Validate input features
            if len(input_features) != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, "
                    f"got {len(input_features)}. "
                    f"Expected features: {self.feature_names}"
                )
            
            # Prepare input: scale and convert to tensor
            input_array = np.array(input_features).reshape(1, -1)
            
            # Scale if scaler is available
            if self.scaler:
                input_scaled = self.scaler.transform(input_array)
            else:
                input_scaled = input_array
                
            input_tensor = torch.FloatTensor(input_scaled)
            
            # Make prediction
            with torch.no_grad():
                probabilities = self.model(input_tensor).numpy()[0]
            
            # Get risk categories
            risk_categories = (
                self.label_encoder.classes_ 
                if hasattr(self.label_encoder, 'classes_') 
                else ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']
            )
            
            # Determine risk level
            risk_index = np.argmax(probabilities)
            risk_level = {
                'label': risk_categories[risk_index],
                'description': f"{risk_categories[risk_index]} Risk Level",
                'probability': float(probabilities[risk_index])
            }
            
            return {
                'risk_level': risk_level,
                'probabilities': probabilities.tolist(),
                'feature_names': self.feature_names
            }
        
        except Exception as e:
            print(f"Prediction error: {e}")
            raise

def load_workflow_config(config_path=None):
    """Load workflow configuration from YAML file"""
    if config_path is None:
        # Define possible paths for the workflow configuration
        possible_paths = [
            os.path.join(parent_dir, 'config', 'workflow_config.yaml'),
            os.path.join(current_dir, 'config', 'workflow_config.yaml'),
            os.path.join(current_dir, 'workflow_config.yaml'),
            'workflow_config.yaml'
        ]
        
        # Find the first existing config path
        config_path = next((path for path in possible_paths if os.path.exists(path)), None)
        
        if config_path is None:
            # If no config file is found, create a default one
            config_path = _create_default_workflow_config()
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading workflow configuration from {config_path}: {e}")
        return {
            'workflow': {
                'name': 'Disaster Risk Prediction',
                'version': '0.1.0'
            }
        }

def _create_default_workflow_config():
    """Create a default workflow configuration file"""
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
    config_dir = os.path.join(parent_dir, 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    # Create default config file
    config_path = os.path.join(config_dir, 'workflow_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f)
    
    return config_path

def setup_logging():
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(parent_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Default logging configuration
    default_logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'file_handler': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': os.path.join(logs_dir, 'app.log'),
                'mode': 'a',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['default', 'file_handler'],
                'level': 'INFO',
                'propagate': True
            },
            'disaster_prediction': {
                'handlers': ['default', 'file_handler'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }

    # Configure logging
    try:
        dictConfig(default_logging_config)
    except Exception as e:
        print(f"Error configuring logging: {e}")
        logging.basicConfig(level=logging.INFO)

def main():
    # Setup logging first
    setup_logging()
    logger = logging.getLogger('disaster_prediction')

    # Load workflow configuration
    workflow_config = load_workflow_config()

    # Streamlit page configuration
    st.set_page_config(
        page_title=workflow_config.get('workflow', {}).get('name', 'Disaster Risk Prediction'),
        page_icon="🌪️",
        layout="wide"
    )
    
    # Sidebar title with dynamic name
    st.sidebar.title(workflow_config.get('workflow', {}).get('name', "Disaster Risk Prediction"))
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigate", 
        ["Risk Prediction", "Global Risk Map", "Workflow Configuration"]
    )
    
    # Initialize predictor
    try:
        predictor = DisasterRiskPredictor()
        logger.info("Predictor initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize predictor: {e}")
        logger.error(f"Predictor initialization error: {e}")
        return
    
    # Page routing with error handling
    try:
        if page == "Risk Prediction":
            disaster_risk_prediction_page(predictor)
        elif page == "Global Risk Map":
            generate_risk_map(predictor)
        elif page == "Workflow Configuration":
            st.header("Workflow Configuration")
            st.json(workflow_config)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Page routing error: {e}")

if __name__ == "__main__":
    main()
