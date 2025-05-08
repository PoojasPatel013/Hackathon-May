import streamlit as st
import numpy as np
import logging
from frontend.predict import DisasterRiskPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def disaster_risk_prediction_page(predictor):
    """
    Streamlit page for disaster risk prediction
    
    Args:
        predictor (DisasterRiskPredictor): Initialized predictor instance
    """
    st.title("Disaster Risk Prediction")
    
    # Input feature collection
    st.subheader("Enter Risk Factors")
    
    # Dynamically create input fields based on feature names
    input_features = []
    for feature in predictor.feature_names:
        feature_value = st.number_input(
            f"Enter {feature}", 
            value=0.0, 
            step=0.1, 
            key=feature
        )
        input_features.append(feature_value)
    
    # Prediction button
    if st.button("Predict Risk"):
        try:
            # Make prediction
            prediction = predictor.predict(input_features)
            
            # Display results
            st.subheader("Prediction Results")
            st.write(f"**Risk Level:** {prediction['risk_level']['label']}")
            st.write(f"**Probability:** {prediction['risk_level']['probability']:.2%}")
            
            # Visualize probabilities
            st.subheader("Risk Probabilities")
            risk_categories = predictor.config['predictions'].get('risk_categories', 
                ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']
            )
            
            for category, prob in zip(risk_categories, prediction['probabilities']):
                st.progress(prob)
                st.write(f"{category}: {prob:.2%}")
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            logger.error(f"Prediction error: {e}")

def main():
    """
    Main function to demonstrate predictor usage
    """
    try:
        # Initialize predictor
        predictor = DisasterRiskPredictor()
        
        # Run the prediction page
        disaster_risk_prediction_page(predictor)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()