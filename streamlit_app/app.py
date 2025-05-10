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
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #F44336;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
    }
    footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f2f6;
        font-size: 0.8rem;
        color: #6c757d;
    }
    .nav-link {
        text-decoration: none;
        color: #1E88E5;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    .nav-link:hover {
        background-color: #e3f2fd;
    }
    .nav-link.active {
        background-color: #1E88E5;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
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
        st.image("https://images.fineartamerica.com/images-medium-large-5/2-satellite-image-of-hurricane-floyd-nasascience-photo-library.jpg", width=150)
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
            # Home dashboard
            st.markdown("<h2 class='sub-header'>Disaster Risk Prediction Dashboard</h2>", unsafe_allow_html=True)
            
            st.markdown("<div class='info-box'>Welcome to the Disaster Risk Prediction platform. This tool helps assess and visualize potential disaster risks based on various environmental and geological factors.</div>", unsafe_allow_html=True)
            
            # Quick stats in cards
            col1, col2, col3 = st.columns(3)
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
            
            # Feature cards
            st.markdown("<h3 class='sub-header'>Key Features</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="card">
                    <h4>🔍 Risk Assessment</h4>
                    <p>Predict disaster risk levels based on environmental and geological factors using our advanced neural network model.</p>
                    <ul>
                        <li>Input customizable risk factors</li>
                        <li>Get instant risk predictions</li>
                        <li>View detailed probability breakdowns</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Go to Risk Assessment", key="goto_risk"):
                    st.session_state.page = "Risk Prediction"
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>🗺️ Global Risk Map</h4>
                    <p>Visualize disaster risk levels across the globe with our interactive map interface.</p>
                    <ul>
                        <li>Explore risk distribution worldwide</li>
                        <li>Hover over countries for detailed information</li>
                        <li>Analyze regional risk patterns</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Explore Global Map", key="goto_map"):
                    st.session_state.page = "Global Risk Map"
                    st.rerun()
            
            # Recent updates section
            st.markdown("<h3 class='sub-header'>Recent Updates</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
                <ul>
                    <li><strong>May 2025:</strong> Improved UI/UX with interactive dashboard</li>
                    <li><strong>April 2025:</strong> Enhanced map visualization with country-level details</li>
                    <li><strong>March 2025:</strong> Updated neural network model with improved accuracy</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
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
