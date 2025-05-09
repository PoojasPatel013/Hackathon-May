import os
import sys
import streamlit as st
import yaml
import logging
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

# Import components using direct imports
from streamlit_app.components.risk_assessment import disaster_risk_prediction_page
from streamlit_app.components.map_visualization import generate_risk_map
from streamlit_app.predictor import DisasterRiskPredictor  # Updated import path

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
