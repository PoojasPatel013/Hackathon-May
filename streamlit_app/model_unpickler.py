import pickle
import torch
import torch.nn as nn
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the model class exactly as it was when the model was saved
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

def load_model_with_custom_unpickler(model_path):
    """
    Load a model using a custom unpickler that can handle the DisasterRiskNetwork class
    
    Args:
        model_path (str): Path to the model pickle file
        
    Returns:
        dict: Dictionary containing model and related components
    """
    try:
        logger.info(f"Loading model with custom unpickler from: {model_path}")
        
        # Add the current module to sys.modules under the name "__main__"
        # This helps with unpickling classes that were defined in __main__
        sys.modules["__main__"] = sys.modules[__name__]
        
        # Open the pickle file with our custom unpickler
        with open(model_path, 'rb') as f:
            saved_data = CustomUnpickler(f).load()
        
        logger.info("Model loaded successfully with custom unpickler")
        return saved_data
    
    except Exception as e:
        logger.error(f"Error loading model with custom unpickler: {e}")
        raise
