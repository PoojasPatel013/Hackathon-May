import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import numpy as np
import logging
import requests
import tempfile
import zipfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_natural_earth_data():
    """
    Download Natural Earth data for world map
    
    Returns:
        geopandas.GeoDataFrame: World map data
    """
    try:
        # URL for Natural Earth low resolution countries
        url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"
        
        # Create a temporary directory to store the downloaded data
        temp_dir = tempfile.mkdtemp()
        
        # Log the download attempt
        logger.info(f"Downloading Natural Earth data from {url}")
        
        # Download the zip file
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Failed to download data: HTTP {response.status_code}")
            return None
        
        # Extract the zip file
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(temp_dir)
        
        # Find the shapefile
        shapefile_path = None
        for file in os.listdir(temp_dir):
            if file.endswith(".shp"):
                shapefile_path = os.path.join(temp_dir, file)
                break
        
        if not shapefile_path:
            logger.error("No shapefile found in the downloaded data")
            return None
        
        # Load the shapefile
        world = gpd.read_file(shapefile_path)
        logger.info("Natural Earth data loaded successfully")
        return world
    
    except Exception as e:
        logger.error(f"Error downloading Natural Earth data: {e}")
        return None

def load_world_map():
    """
    Load world map data with risk zones
    
    Returns:
        geopandas.GeoDataFrame: World map data
    """
    try:
        # First, try to load from a local cache if it exists
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'world_map.geojson')
        
        if os.path.exists(cache_file):
            logger.info(f"Loading world map from cache: {cache_file}")
            return gpd.read_file(cache_file)
        
        # If no cache exists, download the data
        world = download_natural_earth_data()
        
        # If download failed, create a simple world map as fallback
        if world is None:
            logger.warning("Using fallback world map data")
            # Create a very simple world map with just a few major countries
            data = {
                'name': ['United States', 'Canada', 'Mexico', 'Brazil', 'United Kingdom', 
                         'France', 'Germany', 'Russia', 'China', 'India', 'Japan', 'Australia'],
                'pop_est': [331002651, 37742154, 128932753, 212559417, 67886011, 
                           65273511, 83783942, 145934462, 1439323776, 1380004385, 126476461, 25499884],
                'gdp_md_est': [21433226, 1774778, 2463969, 3081000, 2925000, 
                              2716000, 4000000, 4016000, 14343000, 8051000, 5079000, 1248000],
                'geometry': [
                    # Simple polygon representations of countries (very approximate)
                    gpd.points_from_xy([-98.5795], [39.8283])[0].buffer(10),  # USA
                    gpd.points_from_xy([-106.3468], [56.1304])[0].buffer(10),  # Canada
                    gpd.points_from_xy([-102.5528], [23.6345])[0].buffer(5),   # Mexico
                    gpd.points_from_xy([-51.9253], [-14.2350])[0].buffer(8),   # Brazil
                    gpd.points_from_xy([-3.4360], [55.3781])[0].buffer(2),     # UK
                    gpd.points_from_xy([2.2137], [46.2276])[0].buffer(3),      # France
                    gpd.points_from_xy([10.4515], [51.1657])[0].buffer(3),     # Germany
                    gpd.points_from_xy([105.3188], [61.5240])[0].buffer(15),   # Russia
                    gpd.points_from_xy([104.1954], [35.8617])[0].buffer(10),   # China
                    gpd.points_from_xy([78.9629], [20.5937])[0].buffer(8),     # India
                    gpd.points_from_xy([138.2529], [36.2048])[0].buffer(3),    # Japan
                    gpd.points_from_xy([133.7751], [-25.2744])[0].buffer(8)    # Australia
                ]
            }
            world = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        # Cache the data for future use
        if world is not None:
            logger.info(f"Caching world map to: {cache_file}")
            world.to_file(cache_file, driver='GeoJSON')
        
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
        with st.spinner("Loading world map data..."):
            world = load_world_map()
            
        if world is None:
            st.error("Could not load world map. Please check your internet connection or try again later.")
            return
        
        # Display info about the map
        st.info("This map shows simulated disaster risk levels for countries around the world. The risk levels are generated based on population, GDP, and random environmental factors.")
        
        # Generate risk predictions for each country
        with st.spinner("Generating risk predictions..."):
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
                    # Use population and GDP if available, otherwise use random values
                    pop = country_data.get('pop_est', np.random.uniform(1e6, 1e9))
                    gdp = country_data.get('gdp_md_est', np.random.uniform(1e3, 2e4))
                    
                    # Scale population and GDP to reasonable ranges
                    pop_scaled = min(10, pop / 1e8)  # Population in hundreds of millions, capped at 10
                    gdp_scaled = min(10, gdp / 1e6)  # GDP in trillions, capped at 10
                    
                    # Generate mock features
                    mock_features = [
                        pop_scaled,                    # Population (scaled)
                        gdp_scaled,                    # GDP (scaled)
                        np.random.uniform(0, 10),      # Random environmental factor
                        np.random.uniform(0, 5),       # Random geological factor
                        np.random.uniform(0, 3)        # Random climate factor
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
                    logger.error(f"Risk prediction error for {country_data.get('name', 'unknown')}: {e}")
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
                hover_data=['Risk'],
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
        st.info("Please try the Risk Prediction page instead, which doesn't require map data.")
