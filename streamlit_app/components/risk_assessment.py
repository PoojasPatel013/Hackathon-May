import streamlit as st
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Risk level descriptions and recommendations
RISK_DESCRIPTIONS = {
    'Very Low Risk': {
        'description': 'The probability of a significant disaster is minimal. Normal preparedness measures are sufficient.',
        'color': '#4CAF50',  # Green
        'recommendations': [
            'Maintain awareness of local emergency procedures',
            'Keep basic emergency supplies on hand',
            'Stay informed about weather conditions'
        ]
    },
    'Low Risk': {
        'description': 'There is a small chance of a disaster occurring. Basic precautionary measures are advised.',
        'color': '#8BC34A',  # Light Green
        'recommendations': [
            'Review emergency plans with family/organization',
            'Check emergency supplies periodically',
            'Be aware of evacuation routes',
            'Monitor local weather alerts'
        ]
    },
    'Moderate Risk': {
        'description': 'There is a notable chance of a disaster. Enhanced preparedness measures should be considered.',
        'color': '#FFC107',  # Amber
        'recommendations': [
            'Prepare emergency kits for home and vehicles',
            'Create a communication plan with family/colleagues',
            'Secure loose objects that could become hazards',
            'Consider potential evacuation needs',
            'Stay updated with official emergency channels'
        ]
    },
    'High Risk': {
        'description': 'There is a significant probability of a disaster. Immediate preparedness actions are recommended.',
        'color': '#FF9800',  # Orange
        'recommendations': [
            'Activate emergency response plans',
            'Secure valuable possessions and important documents',
            'Prepare for possible evacuation',
            'Check on vulnerable neighbors or family members',
            'Follow official guidance closely',
            'Consider reinforcing structures if time permits'
        ]
    },
    'Extreme Risk': {
        'description': 'A disaster is highly likely or imminent. Urgent protective actions should be taken.',
        'color': '#F44336',  # Red
        'recommendations': [
            'Evacuate if instructed by authorities',
            'Move to designated shelters if evacuation is not possible',
            'Implement all emergency protocols immediately',
            'Maintain constant communication with emergency services',
            'Assist others who may need help if safe to do so',
            'Do not attempt to travel through affected areas'
        ]
    }
}

# Feature descriptions and typical ranges
FEATURE_INFO = {
    'Magnitude': {
        'description': 'The strength or intensity of the event (e.g., earthquake magnitude)',
        'min': 6.5,
        'max': 9.1,
        'default': 7.0,
        'step': 0.1,
        'help': 'For earthquakes, the Richter scale typically ranges from 6.5-9.1 in this dataset, with 7+ considered major.'
    },
    'Depth': {
        'description': 'The depth of the event below the surface (e.g., earthquake hypocenter depth in km)',
        'min': 2.7,
        'max': 670.81,
        'default': 74.6,
        'step': 1.0,
        'help': 'Earthquake depths typically range from shallow (0-70km) to deep (300-700km).'
    },
    'Wind Speed': {
        'description': 'The speed of wind in the area (in knots)',
        'min': 10.0,
        'max': 165.0,
        'default': 54.7,
        'step': 5.0,
        'help': 'Hurricane categories range from 1 (64-82 knots) to 5 (>137 knots).'
    },
    'Tsunami Intensity': {
        'description': 'The intensity of potential tsunami waves (scale -4 to 9)',
        'min': -4.14,
        'max': 9.0,
        'default': 1.4,
        'step': 0.1,
        'help': 'Tsunami intensity scale where negative values indicate minimal impact and 9 is extreme devastation.'
    },
    'Significance': {
        'description': 'The overall significance or impact potential of the event (scale 0-3000)',
        'min': 650.0,
        'max': 2910.0,
        'default': 848.0,
        'step': 10.0,
        'help': 'A combined measure of the event\'s potential impact on human life and infrastructure.'
    },
    'MonsoonIntensity': {
        'description': 'The intensity of monsoon rainfall (scale 0-16)',
        'min': 0.0,
        'max': 16.0,
        'default': 5.0,
        'step': 1.0,
        'help': 'Higher values indicate stronger monsoon conditions that may contribute to flooding.'
    },
    'Deforestation': {
        'description': 'Level of deforestation in the area (scale 0-17)',
        'min': 0.0,
        'max': 17.0,
        'default': 5.0,
        'step': 1.0,
        'help': 'Higher values indicate more severe deforestation that may contribute to landslides and flooding.'
    },
    'FFMC': {
        'description': 'Fine Fuel Moisture Code - indicator of ease of ignition (scale 18-96)',
        'min': 18.7,
        'max': 96.2,
        'default': 90.6,
        'step': 0.1,
        'help': 'Higher values indicate drier fine fuels and increased fire danger.'
    },
    'DMC': {
        'description': 'Duff Moisture Code - moisture of loosely compacted organic layers (scale 1-300)',
        'min': 1.1,
        'max': 291.3,
        'default': 110.9,
        'step': 1.0,
        'help': 'Higher values indicate drier duff layers and increased fire danger.'
    },
    'DC': {
        'description': 'Drought Code - moisture content of deep organic layers (scale 7-900)',
        'min': 7.9,
        'max': 860.6,
        'default': 547.9,
        'step': 10.0,
        'help': 'Higher values indicate drier deep organic layers and increased fire danger.'
    }
}

def create_gauge_chart(risk_level, probability):
    """
    Create a gauge chart to visualize the risk level
    
    Args:
        risk_level (str): Risk level category
        probability (float): Probability value (0-1)
    
    Returns:
        plotly.graph_objects.Figure: Gauge chart figure
    """
    # Get color based on risk level
    color = RISK_DESCRIPTIONS[risk_level]['color']
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Level: {risk_level}", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#4CAF50'},  # Very Low
                {'range': [20, 40], 'color': '#8BC34A'},  # Low
                {'range': [40, 60], 'color': '#FFC107'},  # Moderate
                {'range': [60, 80], 'color': '#FF9800'},  # High
                {'range': [80, 100], 'color': '#F44336'}  # Extreme
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_probability_chart(probabilities, risk_categories):
    """
    Create a bar chart to visualize risk probabilities
    
    Args:
        probabilities (list): List of probability values
        risk_categories (list): List of risk category names
    
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Get colors for each risk category
    colors = [RISK_DESCRIPTIONS[cat]['color'] for cat in risk_categories]
    
    # Create bar chart
    fig = px.bar(
        x=risk_categories,
        y=[prob * 100 for prob in probabilities],
        labels={'x': 'Risk Category', 'y': 'Probability (%)'},
        title='Risk Probability Distribution',
        color=risk_categories,
        color_discrete_map=dict(zip(risk_categories, colors))
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Risk Category",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def save_prediction_history(input_features, prediction, feature_names):
    """
    Save prediction to session state history
    
    Args:
        input_features (list): Input feature values
        prediction (dict): Prediction results
        feature_names (list): Feature name list
    """
    # Initialize history in session state if it doesn't exist
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Create history entry
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'features': dict(zip(feature_names, input_features)),
        'risk_level': prediction['risk_level']['label'],
        'probability': prediction['risk_level']['probability'],
        'all_probabilities': prediction['probabilities']
    }
    
    # Add to history (limit to last 10 entries)
    st.session_state.prediction_history.insert(0, history_entry)
    if len(st.session_state.prediction_history) > 10:
        st.session_state.prediction_history = st.session_state.prediction_history[:10]

def show_prediction_history():
    """
    Display prediction history in an expandable section
    """
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        with st.expander("📜 Prediction History", expanded=False):
            # Convert history to DataFrame for display
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            # Display as table
            st.dataframe(
                history_df[['timestamp', 'risk_level', 'probability']],
                use_container_width=True,
                hide_index=True
            )
            
            # Option to clear history
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.success("History cleared!")
                # Replace experimental_rerun with rerun
                st.rerun()

def predict_risk_level(probability):
    """
    Determine risk level based on probability
    
    Args:
        probability (float): Risk probability (0-1)
    
    Returns:
        str: Risk level category
    """
    if probability < 0.2:
        return 'Very Low Risk'
    elif probability < 0.4:
        return 'Low Risk'
    elif probability < 0.6:
        return 'Moderate Risk'
    elif probability < 0.8:
        return 'High Risk'
    else:
        return 'Extreme Risk'

def disaster_risk_prediction_page(predictor):
    """
    Streamlit page for disaster risk prediction
    
    Args:
        predictor (DisasterRiskPredictor): Initialized predictor instance
    """
    # Page header with icon
    st.markdown("<h1 style='text-align: center; color: beige;'>🔍 Disaster Risk Assessment</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <p style='color: #1E293B; margin: 0;'>This tool helps assess potential disaster risks based on environmental and geological factors. 
        Enter the values for each factor below and click "Predict Risk" to get an assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<h3 style='color: beige;'>Enter Risk Factors</h3>", unsafe_allow_html=True)
        
        # Create a form for input
        with st.form("risk_assessment_form"):
            # Dynamically create input fields based on feature names with improved UI
            input_features = []
            
            for feature in predictor.feature_names:
                # Get feature info or use defaults
                feature_info = FEATURE_INFO.get(feature, {
                    'description': f'Value for {feature}',
                    'min': 0.0,
                    'max': 10.0,
                    'default': 5.0,
                    'step': 0.1,
                    'help': f'Enter a value for {feature}'
                })
                
                # Create expandable section with feature info
                with st.expander(f"{feature} ({feature_info['min']} - {feature_info['max']})", expanded=True):
                    # Description
                    st.markdown(f"<p style='margin-bottom: 10px; color: beige;'>{feature_info['description']}</p>", unsafe_allow_html=True)
                    
                    # Slider for input
                    value = st.slider(
                        f"Select value for {feature}",
                        min_value=feature_info['min'],
                        max_value=feature_info['max'],
                        value=feature_info['default'],
                        step=feature_info['step'],
                        help=feature_info['help'],
                        key=f"slider_{feature}"
                    )
                    
                    input_features.append(value)
            
            # Prediction button with improved styling
            submit_button = st.form_submit_button(
                "🔮 Predict Risk",
                help="Click to calculate the disaster risk based on the provided factors"
            )
    
    with col2:
        # Show preset scenarios in the sidebar
        st.markdown("<h3 style='color: beige;'>Preset Scenarios</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: beige;'>Select a scenario to automatically fill the form:</p>", unsafe_allow_html=True)
        
        # Define preset scenarios
        scenarios = {
            "Major Earthquake": [7.5, 15.0, 0.0, 8.0, 9.0],
            "Category 3 Hurricane": [0.0, 0.0, 180.0, 0.0, 7.0],
            "Moderate Tsunami Risk": [6.8, 10.0, 0.0, 5.0, 6.0],
            "Minor Event": [3.0, 50.0, 30.0, 0.0, 2.0],
            "Extreme Combined Event": [8.0, 5.0, 250.0, 9.0, 10.0]
        }
        
        # Create buttons for each scenario
        for scenario_name, scenario_values in scenarios.items():
            if st.button(scenario_name, key=f"scenario_{scenario_name}"):
                try:
                    # Update session state with scenario values
                    for i, feature in enumerate(predictor.feature_names):
                        if i < len(scenario_values):
                            st.session_state[f"slider_{feature}"] = scenario_values[i]
                    st.rerun()
                except Exception as e:
                    st.error(f"Error applying scenario: {e}")
                    logger.error(f"Error applying scenario: {e}")
        
        # Information box
        st.markdown("""
        <div style='background-color: #fff8e1; padding: 15px; border-radius: 5px; margin-top: 20px;'>
            <h4 style='margin-top: 0; color: #1E293B;'>How to use this tool:</h4>
            <ol style='color: #1E293B;'>
                <li>Enter values for each risk factor</li>
                <li>Or select a preset scenario</li>
                <li>Click "Predict Risk" to get results</li>
                <li>Review the risk assessment and recommendations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Process prediction if form is submitted
    if submit_button:
        try:
            with st.spinner("Calculating risk prediction..."):
                # Make prediction
                prediction = predictor.predict(input_features)

                # Update risk level based on probability thresholds
                prediction['risk_level']['label'] = predict_risk_level(prediction['risk_level']['probability'])
                
                # Save to history
                save_prediction_history(input_features, prediction, predictor.feature_names)
                
                # Get risk level and info
                risk_level = prediction['risk_level']['label']
                risk_info = RISK_DESCRIPTIONS[risk_level]
                
                # Display results in a visually appealing way
                st.markdown("<h2 style='text-align: center; margin-top: 30px; color: #1E293B;'>Prediction Results</h2>", unsafe_allow_html=True)
                
                # Risk gauge
                st.plotly_chart(
                    create_gauge_chart(risk_level, prediction['risk_level']['probability']),
                    use_container_width=True
                )
                
                # Risk description and recommendations
                st.markdown(f"""
                <div style='background-color: {risk_info['color']}22; padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid {risk_info['color']};'>
                    <h3 style='margin-top: 0; color: {risk_info['color']};'>{risk_level}</h3>
                    <p style='color: #1E293B;'><strong>Description:</strong> {risk_info['description']}</p>
                    <p style='color: #1E293B;'><strong>Confidence:</strong> {prediction['risk_level']['probability']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("<h3 style='color: #1E293B;'>Risk Probability Distribution</h3>", unsafe_allow_html=True)
                st.plotly_chart(
                    create_probability_chart(
                        prediction['probabilities'],
                        ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']
                    ),
                    use_container_width=True
                )
                
                # Recommendations
                st.markdown(f"""
                <div style='background-color: #e8f5e9; padding: 20px; border-radius: 5px; margin-top: 20px; border-left: 5px solid #4CAF50;'>
                    <h3 style='margin-top: 0; color: #2E7D32;'>Recommended Actions</h3>
                    <ul style='color: #1E293B;'>
                        {"".join([f"<li>{rec}</li>" for rec in risk_info['recommendations']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Input summary
                with st.expander("Input Summary", expanded=False):
                    # Create a DataFrame for display
                    input_df = pd.DataFrame({
                        'Feature': predictor.feature_names,
                        'Value': input_features
                    })
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            logger.error(f"Prediction error: {e}")
    
    # Show prediction history
    show_prediction_history()
    
    # Additional information section
    with st.expander("ℹ️ About Risk Assessment", expanded=False):
        # Use separate markdown calls for each section to ensure proper rendering
        st.markdown("<h3 style='color: white;'>Understanding Risk Factors</h3>", unsafe_allow_html=True)
        
        st.markdown("<p style='color: white;'>The disaster risk prediction model uses the following factors to assess risk:</p>", unsafe_allow_html=True)
        
        st.markdown("""
        <ul style='color: white;'>
            <li><strong>Magnitude</strong>: The strength or intensity of the event (e.g., earthquake magnitude)</li>
            <li><strong>Depth</strong>: The depth of the event below the surface (e.g., earthquake hypocenter depth)</li>
            <li><strong>Wind Speed</strong>: The speed of wind in the area</li>
            <li><strong>Tsunami Intensity</strong>: The intensity of potential tsunami waves</li>
            <li><strong>Significance</strong>: The overall significance or impact potential of the event</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: white;'>Risk Levels Explained</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <ul style='color: white;'>
            <li><strong>Very Low Risk</strong>: Minimal probability of a significant disaster</li>
            <li><strong>Low Risk</strong>: Small chance of a disaster occurring</li>
            <li><strong>Moderate Risk</strong>: Notable chance of a disaster</li>
            <li><strong>High Risk</strong>: Significant probability of a disaster</li>
            <li><strong>Extreme Risk</strong>: A disaster is highly likely or imminent</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: white;'>Model Information</h3>", unsafe_allow_html=True)
        
        st.markdown("<p style='color: white;'>This risk assessment uses a neural network model trained on historical disaster data. The model analyzes the input factors to predict the probability of different risk levels.</p>", unsafe_allow_html=True)
