import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import logging
import requests
import tempfile
import zipfile
import io
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Risk level colors
RISK_COLORS = {
    'Very Low Risk': '#4CAF50',  # Green
    'Low Risk': '#8BC34A',       # Light Green
    'Moderate Risk': '#FFC107',  # Amber
    'High Risk': '#FF9800',      # Orange
    'Extreme Risk': '#F44336',   # Red
    'Unknown Risk': '#9E9E9E'    # Gray
}

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

def predict_country_risk(country_data, predictor):
    """
    Predict risk for a given country by generating features based on JSON data
    
    Args:
        country_data (geopandas.GeoSeries): Country data
        predictor (DisasterRiskPredictor): Initialized predictor instance
    
    Returns:
        dict: Risk prediction results
    """
    try:
        # Get country name
        country_name = country_data.get('name', 'Unknown')
        
        # Generate features based on country location and JSON data
        lat = country_data.get('latitude', 0)
        lon = country_data.get('longitude', 0)
        
        # Base risk on latitude (tropical regions have higher monsoon/storm risk)
        tropical_factor = max(0, 10 - abs(lat) / 2.3) if -23.5 <= lat <= 23.5 else 0
        
        # Coastal risk (simplified - based on longitude variance in region)
        coastal_factor = min(10, abs(lon) / 18)
        
        # Population factor
        pop = country_data.get('pop_est', np.random.uniform(1e6, 1e9))
        pop_scaled = min(10, pop / 1e8)
        
        # Generate features based on JSON data ranges
        features = {
            'Magnitude': np.random.uniform(6.5, 7.5) if abs(lat) > 30 else np.random.uniform(7.0, 9.1),
            'Depth': np.random.uniform(10, 300) if abs(lat) > 30 else np.random.uniform(5, 100),
            'Wind Speed': np.random.uniform(10, 80) if abs(lat) > 30 else np.random.uniform(50, 165),
            'Tsunami Intensity': 0 if abs(lon) < 90 else np.random.uniform(0, 5),
            'Significance': np.random.uniform(650, 1500) if pop_scaled < 5 else np.random.uniform(1000, 2910),
            'MonsoonIntensity': np.random.uniform(0, 5) if abs(lat) > 30 else np.random.uniform(5, 16),
            'Deforestation': np.random.uniform(0, 17),
            'FFMC': np.random.uniform(70, 96.2) if abs(lat) < 40 else np.random.uniform(18.7, 80),
            'DMC': np.random.uniform(50, 291.3) if abs(lat) < 40 else np.random.uniform(1.1, 100),
            'DC': np.random.uniform(300, 860.6) if abs(lat) < 40 else np.random.uniform(7.9, 400)
        }
        
        # Select features that match predictor's expected input
        input_features = []
        for feature_name in predictor.feature_names:
            if feature_name in features:
                input_features.append(features[feature_name])
            else:
                # Use a default value if feature not found
                input_features.append(5.0)
        
        # Ensure input_features matches expected length
        if len(input_features) > len(predictor.feature_names):
            input_features = input_features[:len(predictor.feature_names)]
        elif len(input_features) < len(predictor.feature_names):
            input_features.extend([5.0] * (len(predictor.feature_names) - len(input_features)))
        
        # Predict risk
        prediction = predictor.predict(input_features)
        
        # Determine risk level based on probability thresholds
        probability = prediction['risk_level']['probability']
        if probability < 0.2:
            risk_level = 'Very Low Risk'
        elif probability < 0.4:
            risk_level = 'Low Risk'
        elif probability < 0.6:
            risk_level = 'Moderate Risk'
        elif probability < 0.8:
            risk_level = 'High Risk'
        else:
            risk_level = 'Extreme Risk'
        
        # Update prediction with risk level
        prediction['risk_level']['label'] = risk_level
        
        # Add additional information to the prediction
        prediction['country'] = country_name
        prediction['population'] = pop
        prediction['gdp'] = country_data.get('gdp_md_est', 0)
        prediction['input_features'] = dict(zip(predictor.feature_names, input_features))
        
        return prediction
    
    except Exception as e:
        logger.error(f"Risk prediction error for {country_data.get('name', 'unknown')}: {e}")
        return {
            'risk_level': {'label': 'Unknown Risk', 'probability': 0.0},
            'country': country_data.get('name', 'Unknown'),
            'population': country_data.get('pop_est', 0),
            'gdp': country_data.get('gdp_md_est', 0)
        }

def create_risk_distribution_chart(risk_counts):
    """
    Create a bar chart for risk distribution
    
    Args:
        risk_counts (pandas.Series): Risk level counts
    
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add bar for each risk level
    for risk_level, count in risk_counts.items():
        fig.add_trace(go.Bar(
            x=[risk_level],
            y=[count],
            name=risk_level,
            marker_color=RISK_COLORS.get(risk_level, '#9E9E9E'),
            text=[count],
            textposition='auto'
        ))
    
    # Update layout
    fig.update_layout(
        title='Global Risk Distribution',
        xaxis_title='Risk Level',
        yaxis_title='Number of Countries',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_risk_map(world_data):
    """
    Create an interactive choropleth map with improved data from JSON
    
    Args:
        world_data (geopandas.GeoDataFrame): World map data with risk levels
    
    Returns:
        plotly.graph_objects.Figure: Choropleth map figure
    """
    # Create choropleth map
    fig = px.choropleth(
        world_data, 
        geojson=world_data.geometry, 
        locations=world_data.index, 
        color='Risk',
        color_discrete_map=RISK_COLORS,
        hover_name='name',
        hover_data={
            'Risk': True,
            'Population': True,
            'GDP': True,
            'Risk_Probability': ':.2%',
            'Primary_Hazard': True
        },
        title='Global Disaster Risk Map',
        projection='natural earth'
    )
    
    # Update map appearance
    fig.update_geos(
        showcoastlines=True, 
        coastlinecolor="Black",
        showland=True, 
        landcolor="lightgray",
        showcountries=True, 
        countrycolor="white",
        showocean=True,
        oceancolor="#E0F7FA",
        showlakes=True,
        lakecolor="#E0F7FA",
        showrivers=True,
        rivercolor="#E0F7FA"
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        title={
            'text': 'Global Disaster Risk Assessment',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        }
    )
    
    return fig

def determine_primary_hazard(lat, lon, features):
    """
    Determine the primary hazard type for a location based on features
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        features (dict): Feature values
    
    Returns:
        str: Primary hazard type
    """
    # Check for earthquake risk
    if features.get('Magnitude', 0) > 7.5:
        return 'Earthquake'
    
    # Check for tsunami risk
    if features.get('Tsunami Intensity', 0) > 3 and abs(lon) > 90:
        return 'Tsunami'
    
    # Check for hurricane/cyclone risk
    if features.get('Wind Speed', 0) > 100 and abs(lat) < 40:
        return 'Hurricane/Cyclone'
    
    # Check for flood risk
    if features.get('MonsoonIntensity', 0) > 10 and abs(lat) < 30:
        return 'Flood'
    
    # Check for wildfire risk
    if features.get('FFMC', 0) > 90 and features.get('DMC', 0) > 200:
        return 'Wildfire'
    
    # Check for volcano risk
    if features.get('Significance', 0) > 2000:
        return 'Volcano'
    
    # Default to general disaster risk
    return 'Multiple Hazards'

def generate_risk_map(predictor):
    """
    Generate and display a global risk map using GeoPandas
    
    Args:
        predictor (DisasterRiskPredictor): Initialized predictor instance
    """
    # Page header with icon
    st.markdown("<h1 style='text-align: center;'>🗺️ Global Disaster Risk Map</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style='background-color: black; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <p>This interactive map shows simulated disaster risk levels for countries around the world. 
        The risk levels are generated based on population, GDP, and environmental factors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Region filter
        regions = ['All Regions', 'North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania']
        selected_region = st.selectbox('Filter by Region', regions)
    
    with col2:
        # Risk level filter
        risk_levels = ['All Risk Levels'] + list(RISK_COLORS.keys())
        selected_risk = st.selectbox('Filter by Risk Level', risk_levels)
    
    with col3:
        # Map view options
        map_view = st.selectbox('Map View', ['Risk Levels', 'Population Density', 'GDP'])
    
    # Add date selector for simulation
    simulation_date = st.date_input(
        "Simulation Date",
        value=datetime.now().date(),
        help="Select a date for the risk simulation"
    )
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load world map
        status_text.text("Loading world map data...")
        progress_bar.progress(10)
        
        world = load_world_map()
        
        if world is None:
            st.error("Could not load world map. Please check your internet connection or try again later.")
            return
        
        progress_bar.progress(30)
        status_text.text("Generating risk predictions...")
        
        # Generate risk predictions for each country
        predictions = []
        total_countries = len(world)
        
        for i, (_, country) in enumerate(world.iterrows()):
            # Update progress
            progress_value = 30 + int((i / total_countries) * 50)
            progress_bar.progress(progress_value)
            
            # Predict risk for country
            prediction = predict_country_risk(country, predictor)
            predictions.append(prediction)
            
            # Update status every 10 countries
            if i % 10 == 0 or i == total_countries - 1:
                status_text.text(f"Analyzing country {i+1} of {total_countries}: {country.get('name', 'Unknown')}")
        
        # Add primary hazard type to world map
        world['Primary_Hazard'] = [
            determine_primary_hazard(
                country.get('latitude', 0),
                country.get('longitude', 0),
                p['input_features']
            ) for p, (_, country) in zip(predictions, world.iterrows())
        ]
        
        progress_bar.progress(80)
        status_text.text("Creating visualization...")
        
        # Add prediction data to world map
        world['Risk'] = [p['risk_level']['label'] for p in predictions]
        world['Population'] = [p['population'] for p in predictions]
        world['GDP'] = [p['gdp'] for p in predictions]
        world['Risk_Probability'] = [p['risk_level']['probability'] for p in predictions]
        
        # Apply filters
        filtered_world = world.copy()
        
        # Region filter
        if selected_region != 'All Regions':
            # This is a simplified example - in a real app, you would need proper region data
            # For now, we'll use a random assignment for demonstration
            if 'region' not in filtered_world.columns:
                # Assign random regions for demonstration
                regions_list = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania']
                filtered_world['region'] = np.random.choice(regions_list, size=len(filtered_world))
            
            filtered_world = filtered_world[filtered_world['region'] == selected_region]
        
        # Risk level filter
        if selected_risk != 'All Risk Levels':
            filtered_world = filtered_world[filtered_world['Risk'] == selected_risk]
        
        progress_bar.progress(90)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Risk Map", "Statistics", "Country Details"])
        
        with tab1:
            # Display the map
            if len(filtered_world) > 0:
                # Choose color scheme based on map view
                if map_view == 'Risk Levels':
                    fig = create_risk_map(filtered_world)
                elif map_view == 'Population Density':
                    fig = px.choropleth(
                        filtered_world,
                        geojson=filtered_world.geometry,
                        locations=filtered_world.index,
                        color='Population',
                        color_continuous_scale='Viridis',
                        hover_name='name',
                        hover_data=['Risk', 'Population', 'GDP'],
                        title='Global Population Density',
                        projection='natural earth'
                    )
                else:  # GDP view
                    fig = px.choropleth(
                        filtered_world,
                        geojson=filtered_world.geometry,
                        locations=filtered_world.index,
                        color='GDP',
                        color_continuous_scale='Plasma',
                        hover_name='name',
                        hover_data=['Risk', 'Population', 'GDP'],
                        title='Global GDP Distribution',
                        projection='natural earth'
                    )
                
                # Update map appearance
                fig.update_geos(
                    showcoastlines=True,
                    coastlinecolor="Black",
                    showland=True,
                    landcolor="lightgray",
                    showcountries=True,
                    countrycolor="white",
                    showocean=True,
                    oceancolor="#E0F7FA"
                )
                
                # Display the map
                st.plotly_chart(fig, use_container_width=True)
                
                # Map legend and explanation
                st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;'>
        <h4 style='color: #1E293B;'>Map Legend</h4>
        <div style='display: flex; flex-wrap: wrap;'>
            <div style='display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;'>
                <div style='width: 20px; height: 20px; background-color: #4CAF50; margin-right: 5px;'></div>
                <span style='color: #1E293B;'>Very Low Risk</span>
            </div>
            <div style='display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;'>
                <div style='width: 20px; height: 20px; background-color: #8BC34A; margin-right: 5px;'></div>
                <span style='color: #1E293B;'>Low Risk</span>
            </div>
            <div style='display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;'>
                <div style='width: 20px; height: 20px; background-color: #FFC107; margin-right: 5px;'></div>
                <span style='color: #1E293B;'>Moderate Risk</span>
            </div>
            <div style='display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;'>
                <div style='width: 20px; height: 20px; background-color: #FF9800; margin-right: 5px;'></div>
                <span style='color: #1E293B;'>High Risk</span>
            </div>
            <div style='display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;'>
                <div style='width: 20px; height: 20px; background-color: #F44336; margin-right: 5px;'></div>
                <span style='color: #1E293B;'>Extreme Risk</span>
            </div>
            <div style='display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;'>
                <div style='width: 20px; height: 20px; background-color: #9E9E9E; margin-right: 5px;'></div>
                <span style='color: #1E293B;'>Unknown Risk</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
            else:
                st.warning("No countries match the selected filters. Please adjust your filter criteria.")
        
        with tab2:
            # Risk distribution statistics
            st.subheader("Risk Distribution Statistics")
            
            # Create two columns for charts
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                # Risk distribution chart
                risk_counts = filtered_world['Risk'].value_counts()
                fig = create_risk_distribution_chart(risk_counts)
                st.plotly_chart(fig, use_container_width=True)
            
            with stat_col2:
                # Risk by population pie chart
                population_by_risk = filtered_world.groupby('Risk')['Population'].sum()
                
                fig = px.pie(
                    values=population_by_risk.values,
                    names=population_by_risk.index,
                    title='Population by Risk Level',
                    color=population_by_risk.index,
                    color_discrete_map=RISK_COLORS
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            # Create metrics row
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Total Countries",
                    len(filtered_world),
                    delta=f"{len(filtered_world) - len(world)}" if len(filtered_world) != len(world) else None
                )
            
            with metric_col2:
                high_risk_count = len(filtered_world[filtered_world['Risk'].isin(['High Risk', 'Extreme Risk'])])
                st.metric(
                    "High/Extreme Risk Countries",
                    high_risk_count,
                    delta=f"{high_risk_count/len(filtered_world):.1%}" if len(filtered_world) > 0 else None,
                    delta_color="inverse"
                )
            
            with metric_col3:
                total_pop = filtered_world['Population'].sum() / 1e9  # Convert to billions
                st.metric(
                    "Total Population",
                    f"{total_pop:.2f}B"
                )
            
            with metric_col4:
                total_gdp = filtered_world['GDP'].sum() / 1e12  # Convert to trillions
                st.metric(
                    "Total GDP",
                    f"${total_gdp:.2f}T"
                )
        
        with tab3:
            # Country details table
            st.subheader("Country Details")
            
            # Search box for countries
            search_term = st.text_input("Search for a country", "")
            
            # Filter countries by search term
            if search_term:
                country_data = filtered_world[filtered_world['name'].str.contains(search_term, case=False)]
            else:
                country_data = filtered_world
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                ["Name (A-Z)", "Risk Level (High to Low)", "Risk Level (Low to High)", "Population (High to Low)", "GDP (High to Low)"]
            )
            
            # Apply sorting
            if sort_by == "Name (A-Z)":
                country_data = country_data.sort_values('name')
            elif sort_by == "Risk Level (High to Low)":
                risk_order = {
                    'Extreme Risk': 5,
                    'High Risk': 4,
                    'Moderate Risk': 3,
                    'Low Risk': 2,
                    'Very Low Risk': 1,
                    'Unknown Risk': 0
                }
                country_data['risk_order'] = country_data['Risk'].map(risk_order)
                country_data = country_data.sort_values('risk_order', ascending=False)
            elif sort_by == "Risk Level (Low to High)":
                risk_order = {
                    'Extreme Risk': 5,
                    'High Risk': 4,
                    'Moderate Risk': 3,
                    'Low Risk': 2,
                    'Very Low Risk': 1,
                    'Unknown Risk': 0
                }
                country_data['risk_order'] = country_data['Risk'].map(risk_order)
                country_data = country_data.sort_values('risk_order')
            elif sort_by == "Population (High to Low)":
                country_data = country_data.sort_values('Population', ascending=False)
            elif sort_by == "GDP (High to Low)":
                country_data = country_data.sort_values('GDP', ascending=False)
            
            # Display table with formatted data
            display_data = country_data[['name', 'Risk', 'Population', 'GDP', 'Risk_Probability']].copy()
            
            # Format population and GDP
            display_data['Population'] = display_data['Population'].apply(lambda x: f"{x/1e6:.2f}M")
            display_data['GDP'] = display_data['GDP'].apply(lambda x: f"${x/1e9:.2f}B")
            display_data['Risk Probability'] = display_data['Risk_Probability'].apply(lambda x: f"{x:.2%}")
            
            # Rename columns for display
            display_data = display_data.rename(columns={
                'name': 'Country',
                'Risk': 'Risk Level',
                'Risk_Probability': 'Risk Probability'
            })
            
            # Display table
            st.dataframe(
                display_data[['Country', 'Risk Level', 'Population', 'GDP', 'Risk Probability']],
                use_container_width=True,
                hide_index=True
            )
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("Map generation complete!")
        
        # Additional information
        with st.expander("ℹ️ About This Map", expanded=False):
            st.markdown("""
            <div style="color: #1E293B;">
            <h3>Understanding the Global Risk Map</h3>
            
            <p>This interactive map visualizes disaster risk levels across countries worldwide. The risk assessment is based on a combination of factors including:</p>
            
            <ul>
                <li><strong>Population density</strong>: Higher population density can increase vulnerability to disasters</li>
                <li><strong>Economic factors</strong>: GDP and infrastructure development affect resilience</li>
                <li><strong>Environmental conditions</strong>: Geographic location and climate factors</li>
                <li><strong>Historical disaster data</strong>: Patterns of previous disasters in the region</li>
            </ul>
            
            <h3>Data Sources</h3>
            
            <ul>
                <li>Country boundaries: Natural Earth Data</li>
                <li>Population and GDP: World Bank (simulated for this demonstration)</li>
                <li>Risk assessment: Neural network prediction model</li>
            </ul>
            
            <h3>Limitations</h3>
            
            <p>This is a simulation for demonstration purposes. In a real-world application, the risk assessment would be based on:</p>
            
            <ul>
                <li>Real-time environmental monitoring data</li>
                <li>Detailed geological and meteorological information</li>
                <li>Historical disaster records</li>
                <li>Infrastructure and vulnerability assessments</li>
            </ul>
            
            <h3>How to Use This Map</h3>
            
            <ul>
                <li><strong>Zoom and pan</strong>: Navigate the map to focus on specific regions</li>
                <li><strong>Hover over countries</strong>: View detailed risk information</li>
                <li><strong>Use filters</strong>: Focus on specific regions or risk levels</li>
                <li><strong>Switch views</strong>: Toggle between risk, population, and GDP visualizations</li>
                <li><strong>View statistics</strong>: Analyze risk distribution and country details in the tabs</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error generating risk map: {e}")
        logger.error(f"Error generating risk map: {e}")
        st.info("Please try the Risk Prediction page instead, which doesn't require map data.")
