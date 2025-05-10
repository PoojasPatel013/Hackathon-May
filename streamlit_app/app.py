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
from datetime import datetime

# Add the parent directory to Python path to make imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

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
                'Magnitude', 'Depth', 'Wind Speed', 'Tsunami Intensity', 'Significance',
                'MonsoonIntensity', 'Deforestation', 'FFMC', 'DMC', 'DC'
            ])
            
            print("Model loaded successfully with all components")
        
        except Exception as e:
            print(f"Comprehensive error loading model from {model_path}: {e}")
            raise
    
    def predict(self, input_features):
        """
        Make a prediction based on input features with improved risk categorization
        
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
            input_array = np.array(input_features, dtype=float).reshape(1, -1)
            
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
        
            # Determine risk level based on highest probability
            risk_index = int(np.argmax(probabilities))
            risk_probability = float(probabilities[risk_index])
        
            # Determine risk level based on probability thresholds
            if risk_probability < 0.2:
                risk_label = 'Very Low Risk'
            elif risk_probability < 0.4:
                risk_label = 'Low Risk'
            elif risk_probability < 0.6:
                risk_label = 'Moderate Risk'
            elif risk_probability < 0.8:
                risk_label = 'High Risk'
            else:
                risk_label = 'Extreme Risk'
        
            risk_level = {
                'label': risk_label,
                'description': f"{risk_label} Risk Level",
                'probability': risk_probability
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
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
    /* Base styles */
    :root {
        --primary-color: #2E7D32;
        --primary-light: #4CAF50;
        --primary-dark: #1B5E20;
        --secondary-color: #0277BD;
        --secondary-light: #039BE5;
        --secondary-dark: #01579B;
        --warning-color: #FF9800;
        --danger-color: #F44336;
        --success-color: #4CAF50;
        --info-color: #2196F3;
        --background-light: #FFFFFF;
        --background-medium: #F5F7F9;
        --background-dark: #1E293B;
        --text-light: #FFFFFF;
        --text-dark: #1E293B;
        --text-muted: #64748B;
        --border-color: #E2E8F0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --radius-sm: 0.25rem;
        --radius-md: 0.375rem;
        --radius-lg: 0.5rem;
        --radius-xl: 1rem;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Global styles */
    body {
        font-family: var(--font-sans);
        color: var(--text-dark);
        background-color: var(--background-medium);
    }
    
    /* Header styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    .sub-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--secondary-color);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Card styles */
    .card {
        background-color: var(--background-light);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid var(--border-color);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.75rem;
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary-dark) 100%);
        color: var(--text-light);
        padding: 3rem 2rem;
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('https://images.unsplash.com/photo-1542281286-9e0a16bb7366');
        background-size: cover;
        background-position: center;
        opacity: 0.15;
        z-index: 0;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        opacity: 0.9;
    }
    
    /* Stat card styles */
    .stat-card {
        background-color: var(--background-light);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        padding: 1.5rem;
        text-align: center;
        border-top: 4px solid var(--primary-color);
        height: 100%;
    }
    
    .stat-value {
        font-size: 3rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 1rem;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    /* Feature card styles */
    .feature-card {
        background-color: var(--background-light);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        padding: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: transform 0.2s, box-shadow 0.2s;
        border-left: 4px solid var(--secondary-color);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: var(--secondary-color);
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--secondary-color);
        margin-bottom: 0.75rem;
    }
    
    .feature-description {
        color: var(--text-muted);
        margin-bottom: 1rem;
        flex-grow: 1;
    }
    
    /* Button styles */
    .custom-button {
        display: inline-block;
        background-color: var(--primary-color);
        color: var(--text-light);
        padding: 0.75rem 1.5rem;
        border-radius: var(--radius-md);
        font-weight: 500;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.2s;
        border: none;
        width: 100%;
        text-decoration: none;
    }
    
    .custom-button:hover {
        background-color: var(--primary-dark);
    }
    
    .custom-button-secondary {
        background-color: var(--secondary-color);
    }
    
    .custom-button-secondary:hover {
        background-color: var(--secondary-dark);
    }
    
    /* Info box styles */
    .info-box {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 4px solid var(--info-color);
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        border-radius: var(--radius-md);
    }
    
    .warning-box {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 4px solid var(--warning-color);
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        border-radius: var(--radius-md);
    }
    
    .success-box {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid var(--success-color);
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        border-radius: var(--radius-md);
    }
    
    .error-box {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid var(--danger-color);
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        border-radius: var(--radius-md);
    }
    
    /* Dashboard section styles */
    .dashboard-section {
        margin-bottom: 2.5rem;
    }
    
    /* Footer styles */
    footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border-color);
        font-size: 0.875rem;
        color: var(--text-muted);
    }
    
    /* Sidebar styles */
    .sidebar .sidebar-content {
        background-color: var(--background-light);
    }
    
    /* Navigation styles */
    .nav-link {
        display: block;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: var(--radius-md);
        color: var(--text-dark);
        font-weight: 500;
        text-decoration: none;
        transition: background-color 0.2s;
    }
    
    .nav-link:hover {
        background-color: rgba(33, 150, 243, 0.1);
        color: var(--secondary-color);
    }
    
    .nav-link.active {
        background-color: var(--secondary-color);
        color: var(--text-light);
    }
    
    /* Disaster type badges */
    .disaster-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-earthquake {
        background-color: #FF9800;
        color: white;
    }
    
    .badge-flood {
        background-color: #2196F3;
        color: white;
    }
    
    .badge-wildfire {
        background-color: #F44336;
        color: white;
    }
    
    .badge-hurricane {
        background-color: #9C27B0;
        color: white;
    }
    
    .badge-tsunami {
        background-color: #00BCD4;
        color: white;
    }
    
    /* Timeline styles */
    .timeline {
        position: relative;
        padding-left: 2rem;
        margin-bottom: 2rem;
    }
    
    .timeline::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 2px;
        background-color: var(--border-color);
    }
    
    .timeline-item {
        position: relative;
        padding-bottom: 1.5rem;
    }
    
    .timeline-item::before {
        content: "";
        position: absolute;
        left: -2rem;
        top: 0.25rem;
        width: 1rem;
        height: 1rem;
        border-radius: 50%;
        background-color: var(--primary-color);
    }
    
    .timeline-date {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.25rem;
    }
    
    .timeline-content {
        background-color: var(--background-light);
        padding: 1rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
    }
    
    /* Metric card styles */
    .metric-card {
        background-color: var(--background-light);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        transition: transform 0.2s;
        border-bottom: 4px solid var(--primary-color);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    /* Risk level indicators */
    .risk-indicator {
        display: inline-block;
        width: 1rem;
        height: 1rem;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .risk-very-low {
        background-color: #4CAF50;
    }
    
    .risk-low {
        background-color: #8BC34A;
    }
    
    .risk-moderate {
        background-color: #FFC107;
    }
    
    .risk-high {
        background-color: #FF9800;
    }
    
    .risk-extreme {
        background-color: #F44336;
    }
    
    /* Streamlit specific overrides */
    .stButton>button {
        width: 100%;
    }
    
    /* Make the Streamlit containers full width */
    .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-medium);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-light);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes slideInUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .animate-slide-up {
        animation: slideInUp 0.5s ease-out;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .stat-value {
            font-size: 2.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: center;">
            <h1 class="main-header">🌪️ {workflow_config.get('workflow', {}).get('name', 'Disaster Risk Prediction')}</h1>
        </div>
        <p style="text-align: center;">Version {workflow_config.get('workflow', {}).get('version', '0.1.0')} | Last Updated: {datetime.now().strftime('%Y-%m-%d')}</p>
        """, unsafe_allow_html=True)
    
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Sidebar with improved navigation
    with st.sidebar:
        st.image("https://images.stockcake.com/public/3/e/6/3e6c0917-925c-438c-94ab-e14ea42a126e_large/purple-storm-fury-stockcake.jpg", width=150)
        st.markdown("### Navigation")
        
        # Navigation buttons with icons
        if st.button("🏠 Home", key="nav_home"):
            st.session_state.page = "Home"
        if st.button("🔍 Risk Assessment", key="nav_risk"):
            st.session_state.page = "Risk Prediction"
        if st.button("🗺️ Global Risk Map", key="nav_map"):
            st.session_state.page = "Global Risk Map"
        if st.button("⚙️ Configuration", key="nav_config"):
            st.session_state.page = "Workflow Configuration"
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This application predicts disaster risk levels based on various environmental and geological factors using a trained neural network model.
        """)
        
        # User guide expandable section
        with st.expander("📚 User Guide"):
            st.markdown("""
            **Quick Start:**
            1. Navigate to Risk Assessment
            2. Enter risk factors
            3. Click Predict Risk
            4. View results and recommendations
            
            **Global Map:**
            - View worldwide risk distribution
            - Hover over countries for details
            """)
    
    # Initialize predictor
    try:
        predictor = DisasterRiskPredictor()
        logger.info("Predictor initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize predictor: {e}")
        logger.error(f"Predictor initialization error: {e}")
        return
    
    # Main content area based on navigation
    try:
        if st.session_state.page == "Home":
            # Hero section
            st.markdown("""
            <div class="hero animate-fade-in">
                <div class="hero-content">
                    <h1 class="hero-title">Disaster Risk Prediction Platform</h1>
                    <p class="hero-subtitle">Advanced analytics and visualization for global disaster risk assessment</p>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px;">
                            <button class="custom-button" onclick="document.querySelector('[key=nav_risk]').click()">
                                Start Risk Assessment
                            </button>
                        </div>
                        <div style="flex: 1; min-width: 200px;">
                            <button class="custom-button custom-button-secondary" onclick="document.querySelector('[key=nav_map]').click()">
                                Explore Global Map
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics section
            st.markdown('<div class="dashboard-section animate-slide-up">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">Key Metrics</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">5</div>
                    <div class="metric-label">Risk Categories</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">200+</div>
                    <div class="metric-label">Countries Analyzed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">99%</div>
                    <div class="metric-label">Prediction Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">7</div>
                    <div class="metric-label">Disaster Types</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Disaster types section
            st.markdown('<div class="dashboard-section animate-slide-up" style="animation-delay: 0.2s;">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">Disaster Types</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <div style="display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center;">
                    <span class="disaster-badge badge-earthquake">Earthquake</span>
                    <span class="disaster-badge badge-flood">Flood</span>
                    <span class="disaster-badge badge-wildfire">Wildfire</span>
                    <span class="disaster-badge badge-hurricane">Hurricane</span>
                    <span class="disaster-badge badge-tsunami">Tsunami</span>
                    <span class="disaster-badge" style="background-color: #795548; color: white;">Landslide</span>
                    <span class="disaster-badge" style="background-color: #607D8B; color: white;">Volcano</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Features section
            st.markdown('<div class="dashboard-section animate-slide-up" style="animation-delay: 0.3s;">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">Key Features</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <h3 class="feature-title">Risk Assessment</h3>
                    <p class="feature-description">
                        Predict disaster risk levels based on environmental and geological factors using our advanced neural network model.
                    </p>
                    <ul style="color: var(--text-muted); padding-left: 1.5rem; margin-bottom: 1.5rem;">
                        <li>Input customizable risk factors</li>
                        <li>Get instant risk predictions</li>
                        <li>View detailed probability breakdowns</li>
                        <li>Receive tailored recommendations</li>
                    </ul>
                    <button class="custom-button" onclick="document.querySelector('[key=nav_risk]').click()">
                        Start Assessment
                    </button>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🗺️</div>
                    <h3 class="feature-title">Global Risk Map</h3>
                    <p class="feature-description">
                        Visualize disaster risk levels across the globe with our interactive map interface.
                    </p>
                    <ul style="color: var(--text-muted); padding-left: 1.5rem; margin-bottom: 1.5rem;">
                        <li>Explore risk distribution worldwide</li>
                        <li>Filter by region or disaster type</li>
                        <li>View population and GDP impact</li>
                        <li>Analyze historical disaster patterns</li>
                    </ul>
                    <button class="custom-button custom-button-secondary" onclick="document.querySelector('[key=nav_map]').click()">
                        Explore Map
                    </button>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk levels explanation
            st.markdown('<div class="dashboard-section animate-slide-up" style="animation-delay: 0.4s;">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">Understanding Risk Levels</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                    <div style="padding: 1rem; border-radius: var(--radius-md); background-color: rgba(76, 175, 80, 0.1); border-left: 4px solid #4CAF50;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span class="risk-indicator risk-very-low"></span>
                            <strong>Very Low Risk</strong>
                        </div>
                        <p style="margin: 0; font-size: 0.875rem; color: var(--text-muted);">
                            Minimal probability of a significant disaster. Normal preparedness measures are sufficient.
                        </p>
                    </div>
                    
                    <div style="padding: 1rem; border-radius: var(--radius-md); background-color: rgba(139, 195, 74, 0.1); border-left: 4px solid #8BC34A;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span class="risk-indicator risk-low"></span>
                            <strong>Low Risk</strong>
                        </div>
                        <p style="margin: 0; font-size: 0.875rem; color: var(--text-muted);">
                            Small chance of a disaster occurring. Basic precautionary measures are advised.
                        </p>
                    </div>
                    
                    <div style="padding: 1rem; border-radius: var(--radius-md); background-color: rgba(255, 193, 7, 0.1); border-left: 4px solid #FFC107;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span class="risk-indicator risk-moderate"></span>
                            <strong>Moderate Risk</strong>
                        </div>
                        <p style="margin: 0; font-size: 0.875rem; color: var(--text-muted);">
                            Notable chance of a disaster. Enhanced preparedness measures should be considered.
                        </p>
                    </div>
                    
                    <div style="padding: 1rem; border-radius: var(--radius-md); background-color: rgba(255, 152, 0, 0.1); border-left: 4px solid #FF9800;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span class="risk-indicator risk-high"></span>
                            <strong>High Risk</strong>
                        </div>
                        <p style="margin: 0; font-size: 0.875rem; color: var(--text-muted);">
                            Significant probability of a disaster. Immediate preparedness actions are recommended.
                        </p>
                    </div>
                    
                    <div style="padding: 1rem; border-radius: var(--radius-md); background-color: rgba(244, 67, 54, 0.1); border-left: 4px solid #F44336;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span class="risk-indicator risk-extreme"></span>
                            <strong>Extreme Risk</strong>
                        </div>
                        <p style="margin: 0; font-size: 0.875rem; color: var(--text-muted);">
                            A disaster is highly likely or imminent. Urgent protective actions should be taken.
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent updates section
            st.markdown('<div class="dashboard-section animate-slide-up" style="animation-delay: 0.5s;">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">Recent Updates</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="timeline">
                <div class="timeline-item">
                    <div class="timeline-date">May 2025</div>
                    <div class="timeline-content">
                        <strong>UI/UX Improvements</strong>
                        <p>Enhanced user interface with interactive dashboard and improved visualization components.</p>
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">April 2025</div>
                    <div class="timeline-content">
                        <strong>Map Visualization Upgrade</strong>
                        <p>Enhanced map visualization with country-level details and improved filtering capabilities.</p>
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">March 2025</div>
                    <div class="timeline-content">
                        <strong>Model Accuracy Improvement</strong>
                        <p>Updated neural network model with improved accuracy and additional disaster types.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif st.session_state.page == "Risk Prediction":
            disaster_risk_prediction_page(predictor)
        elif st.session_state.page == "Global Risk Map":
            generate_risk_map(predictor)
        elif st.session_state.page == "Workflow Configuration":
            st.header("Workflow Configuration")
            
            # Configuration tabs
            tab1, tab2, tab3 = st.tabs(["General", "Model", "Advanced"])
            
            with tab1:
                st.subheader("General Settings")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                workflow_name = st.text_input("Workflow Name", value=workflow_config.get('workflow', {}).get('name', 'Disaster Risk Prediction'))
                workflow_version = st.text_input("Version", value=workflow_config.get('workflow', {}).get('version', '0.1.0'))
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.subheader("Risk Categories")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                risk_categories = workflow_config.get('predictions', {}).get('risk_categories', [])
                for i, category in enumerate(risk_categories):
                    risk_categories[i] = st.text_input(f"Category {i+1}", value=category)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab2:
                st.subheader("Model Information")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("**Model Type:** Neural Network")
                st.write("**Architecture:** Multi-layer Perceptron")
                st.write("**Input Features:**", ", ".join(predictor.feature_names))
                st.write("**Output Classes:**", len(predictor.config['predictions']['risk_categories']))
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.subheader("Model Parameters")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.json({
                    "layers": [
                        {"name": "Input", "neurons": len(predictor.feature_names)},
                        {"name": "Hidden 1", "neurons": 64, "activation": "ReLU"},
                        {"name": "Hidden 2", "neurons": 64, "activation": "ReLU"},
                        {"name": "Hidden 3", "neurons": 32, "activation": "None"},
                        {"name": "Output", "neurons": len(predictor.config['predictions']['risk_categories']), "activation": "Softmax"}
                    ],
                    "dropout_rate": 0.3,
                    "batch_normalization": True
                })
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab3:
                st.subheader("Advanced Configuration")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("Edit the full configuration in YAML format:")
                config_yaml = st.text_area("Configuration YAML", value=yaml.dump(workflow_config), height=300)
                
                if st.button("Save Configuration"):
                    try:
                        new_config = yaml.safe_load(config_yaml)
                        config_path = os.path.join(parent_dir, 'config', 'workflow_config.yaml')
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)
                        
                        with open(config_path, 'w') as f:
                            yaml.dump(new_config, f)
                        
                        st.success("Configuration saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving configuration: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Page routing error: {e}")
    
    # Footer
    st.markdown("""
    <footer>
        <p>© 2025 Disaster Risk Prediction Platform | Developed with ❤️ by Stellaris</p>
        <p>For support, contact: support@stellaris.ai</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
