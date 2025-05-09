import os
import numpy as np
import torch
import logging
from typing import List, Dict, Any
from streamlit_app.model_loader import load_model, DisasterRiskNetwork

# Configure logging
logger = logging.getLogger(__name__)

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
            # Load model components using the dedicated loader
            components = load_model(model_path)
            
            # Extract components
            self.model = components['model']
            self.scaler = components['scaler']
            self.label_encoder = components['label_encoder']
            self.feature_names = components['feature_names']
            
            logger.info("Model loaded successfully with all components")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, input_features: List[float]) -> Dict[str, Any]:
        """
        Make a prediction based on input features
        
        Args:
            input_features (List[float]): Input feature values
        
        Returns:
            Dict[str, Any]: Prediction results
        """
        # Validate input features
        if len(input_features) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"got {len(input_features)}. "
                f"Expected features: {self.feature_names}"
            )
        
        try:
            # Prepare input: scale and convert to tensor
            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = self.scaler.transform(input_array)
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
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, input_features_list: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple input feature sets
        
        Args:
            input_features_list (List[List[float]]): Multiple sets of input features
        
        Returns:
            List[Dict[str, Any]]: Predictions for each input feature set
        """
        return [self.predict(features) for features in input_features_list]
