import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Define the model class here to ensure it's available when loading the pickle file
class DisasterRiskNetwork(nn.Module):
    """
    Neural network model for disaster risk prediction
    Matches the exact architecture of the saved model
    """
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

def load_model(model_path):
    """
    Load the model from a pickle file
    
    Args:
        model_path (str): Path to the model pickle file
        
    Returns:
        dict: Dictionary containing model and related components
    """
    try:
        logger.info(f"Loading model from: {os.path.abspath(model_path)}")
        
        # Verify file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        # Load pickle file
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Log keys in the pickle file
        logger.info("Keys in pickle file:")
        for key in saved_data.keys():
            logger.info(f"  {key}")
        
        # Create model instance
        model = DisasterRiskNetwork(
            input_dim=saved_data['input_dim'], 
            num_classes=saved_data['num_classes']
        )
        
        # Load model state
        model.load_state_dict(saved_data['model_state'], strict=False)
        model.eval()
        
        # Return all components
        return {
            'model': model,
            'scaler': saved_data['scaler'],
            'label_encoder': saved_data['label_encoder'],
            'feature_names': saved_data['feature_names']
        }
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
