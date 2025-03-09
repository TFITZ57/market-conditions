import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import logging
from src.utils.config import load_config

# Configure logger
logger = logging.getLogger(__name__)

def load_fairfield_geojson():
    """
    Load Fairfield County towns GeoJSON data.
    
    Returns:
        dict: GeoJSON data for Fairfield County towns.
    """
    try:
        # Try to load from assets directory
        geojson_path = os.path.join('assets', 'fairfield_towns.geojson')
        
        if os.path.exists(geojson_path):
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
            logger.info(f"Loaded GeoJSON data from {geojson_path}")
            return geojson_data
        
        # If file doesn't exist, return a simplified version
        # This is a placeholder that would need to be replaced with actual GeoJSON data
        logger.warning(f"GeoJSON file not found at {geojson_path}. Using simplified placeholder.")
        
        # Create simplified placeholder with dummy coordinates
        # In a real implementation, this would be actual GeoJSON data for Fairfield County towns
        config = load_config()
        towns = config["locations"]["towns"]
        
        features = []
        for i, town in enumerate(towns):
            # Create a simple square for each town
            # These are not real coordinates, just placeholders
            x_base = (i % 5) * 0.1
            y_base = (i // 5) * 0.1
            
            coordinates = [[[
                [x_base, y_base],
                [x_base + 0.09, y_base],
                [x_base + 0.09, y_base + 0.09],
                [x_base, y_base + 0.09],
                [x_base, y_base]
            ]]]
            
            feature = {
                "type": "Feature",
                "properties": {
                    "name": town
                },
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": coordinates
                }
            }
            
            features.append(feature)
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return geojson_data
    
    except Exception as e:
        logger.error(f"Error loading GeoJSON data: {str(e)}")
        return None

def create_choropleth_map(df, value_col, geojson=None, title=None, color_scale=None):
    """
    Create a choropleth map for Fairfield County towns.
    
    Args:
        df (pd.DataFrame): DataFrame with town data.
        value_col (str): Column to visualize.
        geojson (dict, optional): GeoJSON data. Defaults to None.
        title (str, optional): Map title. Defaults to None.
        color_scale (str, optional): Color scale for map. Defaults to None.
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object.
    """
    try:
        # Ensure town column exists
        if 'town' not in df.columns:
            logger.error("DataFrame missing 'town' column")
            return None
        
        # Ensure value column exists
        if value_col not in df.columns:
            logger.error(f"DataFrame missing '{value_col}' column")
            return None
        
        # Load GeoJSON if not provided
        if geojson is None:
            geojson = load_fairfield_geojson()
            
            if geojson is None:
                logger.error("Failed to load GeoJSON data")
                return None
        
        # Set default title if not provided
        if title is None:
            title = f"{value_col} by Town"
        
        # Set default color scale if not provided
        if color_scale is None:
            # Choose color scale based on the metric type
            if any(term in value_col.lower() for term in ['price', 'value', 'income', 'affordability']):
                color_scale = 'Blues'
            elif any(term in value_col.lower() for term in ['inventory', 'supply', 'listings']):
                color_scale = 'Greens'
            elif any(term in value_col.lower() for term in ['days', 'time', 'dom']):
                color_scale = 'Reds_r'  # Reversed (lower is better)
            else:
                color_scale = 'Viridis'
        
        # Create choropleth map
        fig = px.choropleth_mapbox(
            df,
            geojson=geojson,
            locations='town',
            featureidkey='properties.name',
            color=value_col,
            color_continuous_scale=color_scale,
            mapbox_style="carto-positron",
            zoom=9,
            center={"lat": 41.2, "lon": -73.4},  # Fairfield County approximate center
            opacity=0.7,
            labels={value_col: value_col},
            title=title
        )
        
        # Update layout
        fig.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title=value_col,
                thicknessmode="pixels",
                thickness=20,
                lenmode="pixels",
                len=300,
                yanchor="top",
                y=1,
                ticks="outside"
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating choropleth map: {str(e)}")
        return None

def create_bubble_map(df, location_col='town', size_col=None, color_col=None, hover_data=None, title=None):
    """
    Create a bubble map for Fairfield County towns.
    
    Args:
        df (pd.DataFrame): DataFrame with town data.
        location_col (str, optional): Column with location names. Defaults to 'town'.
        size_col (str, optional): Column for bubble size. Defaults to None.
        color_col (str, optional): Column for bubble color. Defaults to None.
        hover_data (list, optional): Columns to include in hover data. Defaults to None.
        title (str, optional): Map title. Defaults to None.
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object.
    """
    try:
        # Ensure location column exists
        if location_col not in df.columns:
            logger.error(f"DataFrame missing '{location_col}' column")
            return None
        
        # Ensure we have town coordinates
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            logger.info("Adding town coordinates")
            
            # In a real implementation, these would be actual coordinates for Fairfield County towns
            # This is a simplified version with approximate coordinates
            town_coordinates = {
                "Bridgeport": {"lat": 41.1792, "lon": -73.1894},
                "Danbury": {"lat": 41.3948, "lon": -73.4540},
                "Darien": {"lat": 41.0777, "lon": -73.4687},
                "Easton": {"lat": 41.2536, "lon": -73.2982},
                "Fairfield": {"lat": 41.1408, "lon": -73.2613},
                "Greenwich": {"lat": 41.0262, "lon": -73.6282},
                "Monroe": {"lat": 41.3325, "lon": -73.2073},
                "New Canaan": {"lat": 41.1469, "lon": -73.4942},
                "New Fairfield": {"lat": 41.4667, "lon": -73.4833},
                "Newtown": {"lat": 41.4145, "lon": -73.3037},
                "Norwalk": {"lat": 41.1175, "lon": -73.4083},
                "Redding": {"lat": 41.3023, "lon": -73.3876},
                "Ridgefield": {"lat": 41.2811, "lon": -73.4981},
                "Shelton": {"lat": 41.3165, "lon": -73.0931},
                "Stamford": {"lat": 41.0534, "lon": -73.5387},
                "Stratford": {"lat": 41.1845, "lon": -73.1332},
                "Trumbull": {"lat": 41.2681, "lon": -73.2006},
                "Weston": {"lat": 41.2001, "lon": -73.3802},
                "Westport": {"lat": 41.1414, "lon": -73.3579},
                "Wilton": {"lat": 41.1959, "lon": -73.4379}
            }
            
            # Add coordinates to DataFrame
            df = df.copy()
            df['latitude'] = df[location_col].map(lambda x: town_coordinates.get(x, {}).get('lat', None))
            df['longitude'] = df[location_col].map(lambda x: town_coordinates.get(x, {}).get('lon', None))
            
            # Filter out rows with missing coordinates
            missing_coords = df[(df['latitude'].isna()) | (df['longitude'].isna())]
            if not missing_coords.empty:
                logger.warning(f"Missing coordinates for {len(missing_coords)} rows")
                df = df.dropna(subset=['latitude', 'longitude'])
        
        # Set default title if not provided
        if title is None:
            if color_col and size_col:
                title = f"{color_col} and {size_col} by {location_col}"
            elif color_col:
                title = f"{color_col} by {location_col}"
            elif size_col:
                title = f"{size_col} by {location_col}"
            else:
                title = f"{location_col} Map"
        
        # Set hover data if not provided
        if hover_data is None:
            hover_data = [location_col]
            if size_col:
                hover_data.append(size_col)
            if color_col and color_col != size_col:
                hover_data.append(color_col)
        
        # Create bubble map
        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            hover_name=location_col,
            hover_data=hover_data,
            size=size_col if size_col else None,
            color=color_col if color_col else None,
            color_continuous_scale="Viridis" if color_col else None,
            mapbox_style="carto-positron",
            zoom=9,
            center={"lat": 41.2, "lon": -73.4},  # Fairfield County approximate center
            title=title
        )
        
        # Update layout
        fig.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating bubble map: {str(e)}")
        return None

def create_town_comparison_map(df, metric_col, period_col='date', location_col='town', periods=None):
    """
    Create a map to compare metric changes across towns over time.
    
    Args:
        df (pd.DataFrame): DataFrame with town data.
        metric_col (str): Column with metric values.
        period_col (str, optional): Column with time periods. Defaults to 'date'.
        location_col (str, optional): Column with location names. Defaults to 'town'.
        periods (list, optional): List of periods to compare. Defaults to None (latest two periods).
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object.
    """
    try:
        # Ensure required columns exist
        required_cols = [location_col, period_col, metric_col]
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"DataFrame missing '{col}' column")
                return None
        
        # Convert period column to datetime if it's not already
        if df[period_col].dtype != 'datetime64[ns]':
            df = df.copy()
            df[period_col] = pd.to_datetime(df[period_col])
        
        # Get periods to compare
        if periods is None:
            # Use the two most recent periods
            periods = sorted(df[period_col].unique())[-2:]
        
        # Filter data for the selected periods
        period_data = df[df[period_col].isin(periods)]
        
        # Pivot data to get values for each period by town
        pivot_data = period_data.pivot_table(
            index=location_col,
            columns=period_col,
            values=metric_col,
            aggfunc='mean'
        ).reset_index()
        
        # Calculate absolute and percentage change
        if len(periods) >= 2:
            pivot_data['absolute_change'] = pivot_data[periods[1]] - pivot_data[periods[0]]
            pivot_data['percent_change'] = (pivot_data[periods[1]] / pivot_data[periods[0]] - 1) * 100
        
        # Create a bubble map with percent change
        fig = create_bubble_map(
            pivot_data,
            location_col=location_col,
            size_col='absolute_change',
            color_col='percent_change',
            hover_data=[location_col, periods[0], periods[1], 'absolute_change', 'percent_change'],
            title=f"{metric_col} Change by {location_col} ({periods[0].strftime('%Y-%m') if hasattr(periods[0], 'strftime') else periods[0]} to {periods[1].strftime('%Y-%m') if hasattr(periods[1], 'strftime') else periods[1]})"
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating town comparison map: {str(e)}")
        return None

def create_multi_metric_map(df_dict, location_col='town', metrics=None, map_type='bubble'):
    """
    Create maps for multiple metrics.
    
    Args:
        df_dict (dict): Dictionary of DataFrames keyed by metric.
        location_col (str, optional): Column with location names. Defaults to 'town'.
        metrics (list, optional): List of metrics to include. Defaults to None (all metrics).
        map_type (str, optional): Type of map ('bubble' or 'choropleth'). Defaults to 'bubble'.
    
    Returns:
        dict: Dictionary of Plotly figures keyed by metric.
    """
    try:
        # Set default metrics if not provided
        if metrics is None:
            metrics = list(df_dict.keys())
        
        # Create maps for each metric
        map_dict = {}
        
        for metric in metrics:
            if metric not in df_dict:
                logger.warning(f"Metric '{metric}' not found in data")
                continue
            
            df = df_dict[metric]
            
            # Ensure the DataFrame has the location column
            if location_col not in df.columns:
                logger.warning(f"DataFrame for '{metric}' missing '{location_col}' column")
                continue
            
            # Get the most recent period data
            if 'date' in df.columns:
                latest_date = df['date'].max()
                latest_data = df[df['date'] == latest_date]
            else:
                latest_data = df
            
            # Create map based on the specified type
            if map_type.lower() == 'choropleth':
                fig = create_choropleth_map(
                    latest_data,
                    value_col='value',
                    title=f"{metric} by {location_col}"
                )
            else:  # bubble map
                fig = create_bubble_map(
                    latest_data,
                    location_col=location_col,
                    size_col='value',
                    color_col='value',
                    title=f"{metric} by {location_col}"
                )
            
            if fig is not None:
                map_dict[metric] = fig
        
        return map_dict
    
    except Exception as e:
        logger.error(f"Error creating multi-metric maps: {str(e)}")
        return {}
