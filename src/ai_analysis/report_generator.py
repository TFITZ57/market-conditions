import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from src.ai_analysis.llm_clients import get_llm_client
from src.ai_analysis.prompt_templates import create_full_prompt, create_json_format_instructions
from src.utils.config import load_config

# Configure logger
logger = logging.getLogger(__name__)

def prepare_data_for_analysis(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Prepare data dictionary for LLM analysis.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames keyed by metric.
    
    Returns:
        Dict[str, Any]: Prepared data dictionary.
    """
    try:
        prepared_data = {}
        
        for metric, df in data_dict.items():
            if df is None or df.empty:
                continue
            
            # Convert DataFrame to a more manageable format
            # We'll create a list of records for each town
            if 'town' in df.columns and 'date' in df.columns and 'value' in df.columns:
                # Group by town and create time series
                town_data = {}
                
                for town, town_df in df.groupby('town'):
                    # Sort by date
                    town_df = town_df.sort_values('date')
                    
                    # Create a list of [date, value] pairs
                    time_series = town_df[['date', 'value']].values.tolist()
                    
                    # Convert dates to strings
                    time_series = [[d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else d, v] for d, v in time_series]
                    
                    town_data[town] = time_series
                
                prepared_data[metric] = town_data
            
            # If no town column, just create a simple time series
            elif 'date' in df.columns and 'value' in df.columns:
                # Sort by date
                df = df.sort_values('date')
                
                # Create a list of [date, value] pairs
                time_series = df[['date', 'value']].values.tolist()
                
                # Convert dates to strings
                time_series = [[d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else d, v] for d, v in time_series]
                
                prepared_data[metric] = time_series
            
            # If neither town nor date columns, just use the raw values
            else:
                prepared_data[metric] = df.to_dict(orient='records')
        
        # Add metadata
        prepared_data['_metadata'] = {
            'metrics': list(prepared_data.keys()),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_points': sum(len(v) if isinstance(v, list) else 
                            sum(len(town_data) for town_data in v.values()) if isinstance(v, dict) else 1 
                            for k, v in prepared_data.items() if k != '_metadata')
        }
        
        return prepared_data
    
    except Exception as e:
        logger.error(f"Error preparing data for analysis: {str(e)}")
        return {'error': str(e)}

def calculate_summary_statistics(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Calculate summary statistics for the data.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames keyed by metric.
    
    Returns:
        Dict[str, Any]: Summary statistics.
    """
    try:
        summary_stats = {}
        
        for metric, df in data_dict.items():
            if df is None or df.empty:
                continue
            
            # Skip metadata
            if metric.startswith('_'):
                continue
            
            metric_stats = {}
            
            # Calculate basic statistics if value column exists
            if 'value' in df.columns:
                values = df['value'].dropna()
                
                if not values.empty:
                    metric_stats.update({
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std())
                    })
            
            # Calculate town-level statistics if town column exists
            if 'town' in df.columns and 'value' in df.columns:
                town_stats = {}
                
                for town, town_df in df.groupby('town'):
                    town_values = town_df['value'].dropna()
                    
                    if not town_values.empty:
                        town_stats[town] = {
                            'min': float(town_values.min()),
                            'max': float(town_values.max()),
                            'mean': float(town_values.mean()),
                            'median': float(town_values.median()),
                            'std': float(town_values.std()),
                            'count': int(town_values.count())
                        }
                
                metric_stats['by_town'] = town_stats
            
            # Calculate time-based statistics if date column exists
            if 'date' in df.columns and 'value' in df.columns:
                # Make sure date column is datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Get the most recent data
                latest_date = df['date'].max()
                latest_df = df[df['date'] == latest_date]
                
                if not latest_df.empty and 'value' in latest_df.columns:
                    metric_stats['latest'] = {
                        'date': latest_date.strftime('%Y-%m-%d'),
                        'mean': float(latest_df['value'].mean()) 
                    }
                
                # If we have at least a year of data, calculate year-over-year change
                one_year_ago = latest_date - pd.DateOffset(years=1)
                year_ago_df = df[(df['date'] <= one_year_ago) & (df['date'] >= one_year_ago - pd.DateOffset(days=30))]
                
                if not year_ago_df.empty and 'value' in year_ago_df.columns:
                    year_ago_value = year_ago_df['value'].mean()
                    latest_value = latest_df['value'].mean()
                    
                    yoy_change = (latest_value - year_ago_value) / year_ago_value
                    
                    metric_stats['yoy_change'] = float(yoy_change)
                    metric_stats['yoy_change_pct'] = f"{yoy_change * 100:.2f}%"
            
            summary_stats[metric] = metric_stats
        
        return summary_stats
    
    except Exception as e:
        logger.error(f"Error calculating summary statistics: {str(e)}")
        return {'error': str(e)}

def identify_key_trends(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Identify key trends in the data.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames keyed by metric.
    
    Returns:
        Dict[str, Any]: Key trends.
    """
    try:
        key_trends = {}
        
        for metric, df in data_dict.items():
            if df is None or df.empty:
                continue
            
            # Skip metadata
            if metric.startswith('_'):
                continue
            
            metric_trends = {}
            
            # Identify trends if date column exists
            if 'date' in df.columns and 'value' in df.columns:
                # Make sure date column is datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Sort by date
                df = df.sort_values('date')
                
                # Calculate rolling mean to smooth out noise
                if len(df) >= 3:
                    df['rolling_mean'] = df['value'].rolling(window=3, min_periods=1).mean()
                    
                    # Calculate month-over-month change
                    df['mom_change'] = df['rolling_mean'].pct_change()
                    
                    # Determine trend direction
                    recent_changes = df['mom_change'].dropna().tail(3)
                    
                    if len(recent_changes) > 0:
                        avg_recent_change = recent_changes.mean()
                        
                        if avg_recent_change > 0.03:
                            trend_direction = "strong_increase"
                        elif avg_recent_change > 0.01:
                            trend_direction = "moderate_increase"
                        elif avg_recent_change > -0.01:
                            trend_direction = "stable"
                        elif avg_recent_change > -0.03:
                            trend_direction = "moderate_decrease"
                        else:
                            trend_direction = "strong_decrease"
                        
                        metric_trends['direction'] = trend_direction
                        metric_trends['avg_recent_change'] = float(avg_recent_change)
                
                # Check for seasonal patterns
                if len(df) >= 12 and 'date' in df.columns:
                    # Extract quarter or month
                    df['quarter'] = df['date'].dt.quarter
                    
                    # Calculate average by quarter
                    quarterly_avg = df.groupby('quarter')['value'].mean()
                    
                    # Calculate the coefficient of variation
                    cv = quarterly_avg.std() / quarterly_avg.mean()
                    
                    # If CV is significant, there might be seasonality
                    if cv > 0.1:
                        # Identify the highest and lowest quarters
                        high_quarter = quarterly_avg.idxmax()
                        low_quarter = quarterly_avg.idxmin()
                        
                        metric_trends['seasonality'] = {
                            'detected': True,
                            'high_quarter': int(high_quarter),
                            'low_quarter': int(low_quarter),
                            'variation_coefficient': float(cv)
                        }
                    else:
                        metric_trends['seasonality'] = {
                            'detected': False
                        }
            
            # Identify geographical trends if town column exists
            if 'town' in df.columns and 'value' in df.columns and 'date' in df.columns:
                # Calculate average value by town for the most recent period
                latest_date = df['date'].max()
                latest_df = df[df['date'] == latest_date]
                
                town_avg = latest_df.groupby('town')['value'].mean().sort_values(ascending=False)
                
                # Identify towns with highest and lowest values
                if len(town_avg) > 0:
                    metric_trends['geographic'] = {
                        'highest_towns': town_avg.head(3).index.tolist(),
                        'lowest_towns': town_avg.tail(3).index.tolist()
                    }
                
                # Calculate growth rates by town
                if len(df) > 0 and len(df['date'].unique()) > 1:
                    # Get earliest and latest dates
                    earliest_date = df['date'].min()
                    
                    # Only calculate if we have a meaningful time difference
                    if (latest_date - earliest_date).days > 30:
                        earliest_df = df[df['date'] == earliest_date]
                        
                        # Merge earliest and latest data to calculate growth
                        town_growth = pd.merge(
                            earliest_df.groupby('town')['value'].mean().reset_index().rename(columns={'value': 'start_value'}),
                            latest_df.groupby('town')['value'].mean().reset_index().rename(columns={'value': 'end_value'}),
                            on='town'
                        )
                        
                        # Calculate growth rate
                        town_growth['growth_rate'] = (town_growth['end_value'] - town_growth['start_value']) / town_growth['start_value']
                        
                        # Sort by growth rate
                        town_growth = town_growth.sort_values('growth_rate', ascending=False)
                        
                        if len(town_growth) > 0:
                            metric_trends['geographic_growth'] = {
                                'fastest_growing': town_growth.head(3)['town'].tolist(),
                                'slowest_growing': town_growth.tail(3)['town'].tolist(),
                                'timeframe': f"{earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}"
                            }
            
            key_trends[metric] = metric_trends
        
        return key_trends
    
    except Exception as e:
        logger.error(f"Error identifying key trends: {str(e)}")
        return {'error': str(e)}

def generate_report(data_dict: Dict[str, pd.DataFrame], report_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive market analysis report.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames keyed by metric.
        report_type (str): Type of report to generate.
        parameters (Dict[str, Any], optional): Report generation parameters. Defaults to None.
    
    Returns:
        Dict[str, Any]: Generated report.
    """
    try:
        # Set default parameters if not provided
        if parameters is None:
            parameters = {}
        
        # Extract parameters
        provider = parameters.get('provider', None)
        model = parameters.get('model', None)
        
        # Prepare data for analysis
        prepared_data = prepare_data_for_analysis(data_dict)
        
        # Calculate summary statistics
        summary_stats = calculate_summary_statistics(data_dict)
        
        # Identify key trends
        trends = identify_key_trends(data_dict)
        
        # Combine data for analysis
        analysis_data = {
            'raw_data': prepared_data,
            'summary_statistics': summary_stats,
            'key_trends': trends
        }
        
        # Create full prompt
        prompt = create_full_prompt(report_type, analysis_data, parameters)
        
        # Get LLM client
        llm_client = get_llm_client(provider, model)
        
        # Generate report content
        logger.info(f"Generating {report_type} report...")
        report_content = llm_client.generate(prompt, max_tokens=2000, temperature=0.5)
        
        # Create report metadata
        metadata = {
            'report_type': report_type,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': parameters,
            'data_metrics': list(data_dict.keys()),
            'provider': llm_client.provider,
            'model': llm_client.model
        }
        
        # Assemble full report
        report = {
            'content': report_content,
            'metadata': metadata,
            'data_summary': summary_stats
        }
        
        logger.info(f"Successfully generated {report_type} report")
        return report
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return {
            'content': f"Error generating report: {str(e)}",
            'metadata': {
                'report_type': report_type,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
        }

def generate_structured_report(data_dict: Dict[str, pd.DataFrame], report_type: str, schema: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate a structured market analysis report that conforms to a schema.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames keyed by metric.
        report_type (str): Type of report to generate.
        schema (Dict[str, Any]): JSON schema for the report structure.
        parameters (Dict[str, Any], optional): Report generation parameters. Defaults to None.
    
    Returns:
        Dict[str, Any]: Generated structured report.
    """
    try:
        # Set default parameters if not provided
        if parameters is None:
            parameters = {}
        
        # Extract parameters
        provider = parameters.get('provider', None)
        model = parameters.get('model', None)
        
        # Prepare data for analysis
        prepared_data = prepare_data_for_analysis(data_dict)
        
        # Calculate summary statistics
        summary_stats = calculate_summary_statistics(data_dict)
        
        # Identify key trends
        trends = identify_key_trends(data_dict)
        
        # Combine data for analysis
        analysis_data = {
            'raw_data': prepared_data,
            'summary_statistics': summary_stats,
            'key_trends': trends
        }
        
        # Create schema-specific prompt
        prompt = create_full_prompt(report_type, analysis_data, parameters)
        format_instructions = create_json_format_instructions(schema)
        
        # Get LLM client
        llm_client = get_llm_client(provider, model)
        
        # Generate structured report content
        logger.info(f"Generating structured {report_type} report...")
        structured_content = llm_client.structured_generate(prompt, format_instructions, max_tokens=2000, temperature=0.5)
        
        # Create report metadata
        metadata = {
            'report_type': report_type,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': parameters,
            'data_metrics': list(data_dict.keys()),
            'provider': llm_client.provider,
            'model': llm_client.model,
            'schema': schema
        }
        
        # Add metadata to report
        structured_content['_metadata'] = metadata
        
        logger.info(f"Successfully generated structured {report_type} report")
        return structured_content
    
    except Exception as e:
        logger.error(f"Error generating structured report: {str(e)}")
        return {
            'error': str(e),
            '_metadata': {
                'report_type': report_type,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
        }

def generate_comparative_report(data_dict: Dict[str, pd.DataFrame], locations: List[str], metrics: List[str], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate a report comparing multiple locations.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames keyed by metric.
        locations (List[str]): List of locations to compare.
        metrics (List[str]): List of metrics to include.
        parameters (Dict[str, Any], optional): Report generation parameters. Defaults to None.
    
    Returns:
        Dict[str, Any]: Generated comparison report.
    """
    try:
        # Set default parameters if not provided
        if parameters is None:
            parameters = {}
        
        # Set report type
        report_type = "comparative"
        
        # Filter data by locations and metrics
        filtered_data = {}
        
        for metric, df in data_dict.items():
            if metric not in metrics:
                continue
            
            if df is None or df.empty:
                continue
            
            if 'town' in df.columns:
                # Filter by locations
                filtered_df = df[df['town'].isin(locations)]
                
                if not filtered_df.empty:
                    filtered_data[metric] = filtered_df
            else:
                # Include county-wide metrics
                filtered_data[metric] = df
        
        # Update parameters to include locations and metrics
        parameters['selected_locations'] = locations
        parameters['selected_metrics'] = metrics
        parameters['comparison_focus'] = True
        
        # Generate report
        return generate_report(filtered_data, report_type, parameters)
    
    except Exception as e:
        logger.error(f"Error generating comparative report: {str(e)}")
        return {
            'content': f"Error generating comparative report: {str(e)}",
            'metadata': {
                'report_type': 'comparative',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
        }