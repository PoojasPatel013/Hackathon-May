import os
import sys
import streamlit as st
import yaml
import logging
import json
from logging.config import dictConfig


# Add frontend directory to Python path
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, frontend_dir)

# Explicitly add the current directory and parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try relative import first
try:
    from components.risk_assessment import disaster_risk_prediction_page
    from components.map_visualization import generate_risk_map
    from predict import DisasterRiskPredictor
except ImportError:
    try:
        # Try another relative import strategy
        from streamlit_app.components.risk_assessment import disaster_risk_prediction_page
        from streamlit_app.components.map_visualization import generate_risk_map
        from frontend.predict import DisasterRiskPredictor
    except ImportError:
        # Absolute import as last resort
        from frontend.streamlit_app.components.risk_assessment import disaster_risk_prediction_page
        from frontend.streamlit_app.components.map_visualization import generate_risk_map
        from frontend.predict import DisasterRiskPredictor

# Modify import strategy to handle potential PyTorch import issues
def safe_import(module_path):
    try:
        # Use importlib for more robust importing
        import importlib
        module = importlib.import_module(module_path)
        return module
    except ImportError as e:
        print(f"Import error for {module_path}: {e}")
        return None

# Try importing with multiple strategies
def import_components():
    # Possible import paths
    import_strategies = [
        # Strategy 1: Relative imports
        {
            'risk_assessment': 'components.risk_assessment',
            'map_visualization': 'components.map_visualization',
            'predictor': 'predict'
        },
        # Strategy 2: Streamlit app imports
        {
            'risk_assessment': 'streamlit_app.components.risk_assessment',
            'map_visualization': 'streamlit_app.components.map_visualization',
            'predictor': 'frontend.predict'
        },
        # Strategy 3: Absolute imports
        {
            'risk_assessment': 'frontend.streamlit_app.components.risk_assessment',
            'map_visualization': 'frontend.streamlit_app.components.map_visualization',
            'predictor': 'frontend.predict'
        }
    ]
    
    # Try each import strategy
    for strategy in import_strategies:
        try:
            # Import modules
            risk_assessment = safe_import(strategy['risk_assessment'])
            map_visualization = safe_import(strategy['map_visualization'])
            predictor = safe_import(strategy['predictor'])
            
            # Check if all imports are successful
            if all([risk_assessment, map_visualization, predictor]):
                return (
                    risk_assessment.disaster_risk_prediction_page,
                    map_visualization.generate_risk_map,
                    predictor.DisasterRiskPredictor
                )
        except Exception as e:
            print(f"Import strategy failed: {e}")
            continue
    
    # If all strategies fail
    raise ImportError("Could not import required modules. Please check your project structure and imports.")

# Modify the main import section
try:
    # Add sys.path manipulation for additional import paths
    import sys
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, frontend_dir)
    
    # Attempt to import components
    disaster_risk_prediction_page, generate_risk_map, DisasterRiskPredictor = import_components()
except ImportError as e:
    st.error(f"Critical Import Error: {e}")
    st.error("Unable to load application components. Please check your project configuration.")
    
    # Provide a fallback mechanism
    def dummy_page(predictor):
        st.error("Page not available due to import errors")
    
    disaster_risk_prediction_page = dummy_page
    generate_risk_map = dummy_page
    DisasterRiskPredictor = None


def load_workflow_config(config_path=None):
    if config_path is None:
        # Define multiple possible paths for the workflow configuration
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'workflow_config.yaml'),
            os.path.join(os.path.dirname(__file__), '..', 'config', 'workflow_config.yaml'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workflow_config.yaml'),
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
    """
    Create a default workflow configuration file
    
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
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    # Create default config file
    config_path = os.path.join(config_dir, 'workflow_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f)
    
    return config_path

def setup_logging(config_path=None):
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
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
        logging.config.dictConfig(default_logging_config)
    except Exception as e:
        print(f"Error configuring logging: {e}")
        logging.basicConfig(level=logging.INFO)

def main():
    # Setup logging first
    setup_logging()
    logger = logging.getLogger('disaster_prediction')

    # Load workflow configuration
    workflow_config = load_workflow_config()

    # Streamlit page configuration MUST be the first Streamlit command
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