import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_world_map():
    """
    Load world map data with risk zones
    
    Returns:
        geopandas.GeoDataFrame: World map data
    """
    try:
        # Use built-in GeoPandas world map
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        return world
    except Exception as e:
        st.error(f"Error loading world map: {e}")
        logger.error(f"Error loading world map: {e}")
        return None

def generate_risk_map(predictor):
    """
    Generate and display a global risk map using GeoPandas
    
    Args:
        predictor (DisasterRiskPredictor): Initialized predictor instance
    """
    st.title("Global Disaster Risk Map")
    
    try:
        # Load world map
        world = load_world_map()
        if world is None:
            st.error("Could not load world map.")
            return
        
        # Generate risk predictions for each country
        def predict_country_risk(country_data):
            """
            Predict risk for a given country by generating mock features
            
            Args:
                country_data (geopandas.GeoSeries): Country data
            
            Returns:
                str: Risk level prediction
            """
            try:
                # Generate mock features based on country characteristics
                mock_features = [
                    country_data['pop_est'] / 1e6,  # Population (millions)
                    country_data['gdp_md_est'] / 1e3,  # GDP (billions)
                    np.random.uniform(0, 10),  # Random environmental factor
                    np.random.uniform(0, 5),   # Random geological factor
                    np.random.uniform(0, 3)    # Random climate factor
                ]
                
                # Ensure mock features match predictor's expected input length
                if len(mock_features) > len(predictor.feature_names):
                    mock_features = mock_features[:len(predictor.feature_names)]
                elif len(mock_features) < len(predictor.feature_names):
                    # Pad with zeros if needed
                    mock_features.extend([0] * (len(predictor.feature_names) - len(mock_features)))
                
                # Predict risk
                prediction = predictor.predict(mock_features)
                return prediction['risk_level']['label']
            except Exception as e:
                logger.error(f"Risk prediction error for {country_data['name']}: {e}")
                return 'Unknown Risk'
        
        # Add risk predictions to world map
        world['Risk'] = world.apply(predict_country_risk, axis=1)
        
        # Define risk color mapping
        risk_color_map = {
            'Very Low Risk': '#00FF00',  # Green
            'Low Risk': '#90EE90',        # Light Green
            'Moderate Risk': '#FFFF00',   # Yellow
            'High Risk': '#FFA500',       # Orange
            'Extreme Risk': '#FF0000',    # Red
            'Unknown Risk': '#808080'     # Gray
        }
        
        # Create interactive choropleth map
        fig = px.choropleth(
            world, 
            geojson=world.geometry, 
            locations=world.index, 
            color='Risk',
            color_discrete_map=risk_color_map,
            hover_name='name',
            hover_data=['Risk', 'pop_est', 'gdp_md_est'],
            title='Global Disaster Risk Map',
            projection='natural earth'
        )
        
        fig.update_geos(
            showcoastlines=True, 
            coastlinecolor="Black",
            showland=True, 
            landcolor="lightgray",
            showcountries=True, 
            countrycolor="white"
        )
        
        # Display the map
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk distribution summary
        st.subheader("Global Risk Distribution")
        risk_counts = world['Risk'].value_counts()
        st.bar_chart(risk_counts)
    
    except Exception as e:
        st.error(f"Error generating risk map: {e}")
        logger.error(f"Error generating risk map: {e}")
