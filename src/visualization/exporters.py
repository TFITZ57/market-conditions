"""
Visualization Exporters

This module provides utilities for exporting visualizations.
"""

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging
import io
import base64

logger = logging.getLogger(__name__)


def export_visualizations(figures: Union[List[go.Figure], go.Figure], 
                         format_type: str = 'html', 
                         filename_prefix: str = 'visualization',
                         output_dir: Optional[str] = None) -> List[str]:
    """
    Export visualizations to files.
    
    Args:
        figures: List of Plotly figures or single figure
        format_type: Export format ('html', 'png', 'svg', 'pdf')
        filename_prefix: Prefix for output filenames
        output_dir: Directory to save files to
        
    Returns:
        List of paths to saved files
    """
    # Convert single figure to list
    if not isinstance(figures, list):
        figures = [figures]
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = "exports"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export each figure
    saved_files = []
    for i, fig in enumerate(figures):
        # Create filename
        filename = f"{filename_prefix}_{i+1}_{timestamp}.{format_type}"
        filepath = os.path.join(output_dir, filename)
        
        # Export based on format
        if format_type == 'html':
            fig.write_html(filepath)
        elif format_type == 'png':
            fig.write_image(filepath)
        elif format_type == 'svg':
            fig.write_image(filepath)
        elif format_type == 'pdf':
            fig.write_image(filepath)
        else:
            # Default to HTML
            fig.write_html(filepath)
        
        saved_files.append(filepath)
    
    return saved_files

def create_download_link(data: Union[pd.DataFrame, Dict[str, Any], str], 
                        filename: str, 
                        format_type: str = 'csv') -> str:
    """
    Create a download link for data.
    
    Args:
        data: Data to export (DataFrame, dict, or string)
        filename: Name of the downloaded file
        format_type: Export format ('csv', 'json', 'excel', 'text')
        
    Returns:
        HTML download link
    """
    # Generate file content based on format and data type
    if format_type == 'csv' and isinstance(data, pd.DataFrame):
        # Export DataFrame as CSV
        file_content = data.to_csv(index=False)
        mime_type = "text/csv"
    elif format_type == 'json' and (isinstance(data, dict) or isinstance(data, pd.DataFrame)):
        # Export dict or DataFrame as JSON
        if isinstance(data, pd.DataFrame):
            file_content = data.to_json(orient='records')
        else:
            import json
            file_content = json.dumps(data)
        mime_type = "application/json"
    elif format_type == 'excel' and isinstance(data, pd.DataFrame):
        # Export DataFrame as Excel
        import io
        buffer = io.BytesIO()
        data.to_excel(buffer, index=False)
        file_content = buffer.getvalue()
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif format_type == 'text' or isinstance(data, str):
        # Export as text
        file_content = data if isinstance(data, str) else str(data)
        mime_type = "text/plain"
    else:
        # Default to CSV for DataFrames, text for others
        if isinstance(data, pd.DataFrame):
            file_content = data.to_csv(index=False)
            mime_type = "text/csv"
        else:
            file_content = str(data)
            mime_type = "text/plain"
    
    # Create a download link using HTML
    html = f"""
    <a href="data:{mime_type};charset=utf-8,{file_content}" 
       download="{filename}" 
       style="display: inline-block; padding: 10px 20px; 
              background-color: #4CAF50; color: white; 
              text-align: center; text-decoration: none; 
              border-radius: 5px;">
        Download {filename}
    </a>
    """
    
    return html

def figure_to_image(fig, format='png', width=None, height=None, scale=1):
    """
    Convert a Plotly figure to an image.
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure to convert.
        format (str, optional): Image format ('png', 'jpg', 'svg', 'pdf'). Defaults to 'png'.
        width (int, optional): Image width in pixels. Defaults to None.
        height (int, optional): Image height in pixels. Defaults to None.
        scale (int, optional): Image scale factor. Defaults to 1.
    
    Returns:
        bytes: Image bytes.
    """
    try:
        if fig is None:
            logger.error("No figure provided for export")
            return None
        
        # Set image dimensions if provided
        if width is not None or height is not None:
            fig = fig.update_layout(
                width=width,
                height=height
            )
        
        # Convert figure to image
        img_bytes = fig.to_image(
            format=format,
            scale=scale
        )
        
        return img_bytes
    
    except Exception as e:
        logger.error(f"Error converting figure to image: {str(e)}")
        return None

def figure_to_html(fig, include_plotlyjs=True, full_html=True):
    """
    Convert a Plotly figure to HTML.
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure to convert.
        include_plotlyjs (bool, optional): Whether to include Plotly.js. Defaults to True.
        full_html (bool, optional): Whether to include HTML header/body tags. Defaults to True.
    
    Returns:
        str: HTML string.
    """
    try:
        if fig is None:
            logger.error("No figure provided for export")
            return None
        
        # Convert figure to HTML
        html_str = fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )
        
        return html_str
    
    except Exception as e:
        logger.error(f"Error converting figure to HTML: {str(e)}")
        return None

def dataframe_to_csv(df, index=False):
    """
    Convert a DataFrame to CSV string.
    
    Args:
        df (pd.DataFrame): DataFrame to convert.
        index (bool, optional): Whether to include index. Defaults to False.
    
    Returns:
        str: CSV string.
    """
    try:
        if df is None or df.empty:
            logger.error("No DataFrame provided for export")
            return None
        
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=index)
        csv_str = csv_buffer.getvalue()
        
        return csv_str
    
    except Exception as e:
        logger.error(f"Error converting DataFrame to CSV: {str(e)}")
        return None

def dataframe_to_excel(df, sheet_name='Sheet1', index=False):
    """
    Convert a DataFrame to Excel bytes.
    
    Args:
        df (pd.DataFrame): DataFrame to convert.
        sheet_name (str, optional): Excel sheet name. Defaults to 'Sheet1'.
        index (bool, optional): Whether to include index. Defaults to False.
    
    Returns:
        bytes: Excel bytes.
    """
    try:
        if df is None or df.empty:
            logger.error("No DataFrame provided for export")
            return None
        
        # Convert DataFrame to Excel
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, sheet_name=sheet_name, index=index)
        excel_bytes = excel_buffer.getvalue()
        
        return excel_bytes
    
    except Exception as e:
        logger.error(f"Error converting DataFrame to Excel: {str(e)}")
        return None

def create_dashboard_export(figs, data, title=None, description=None, export_dir=None):
    """
    Create a comprehensive dashboard export with visualizations and data.
    
    Args:
        figs (dict): Dictionary of Plotly figures.
        data (dict): Dictionary of DataFrames.
        title (str, optional): Dashboard title. Defaults to None.
        description (str, optional): Dashboard description. Defaults to None.
        export_dir (str, optional): Export directory. Defaults to None.
    
    Returns:
        str: Path to the exported dashboard HTML file.
    """
    try:
        if not figs and not data:
            logger.error("No figures or data provided for dashboard export")
            return None
        
        # Set default title and description
        if title is None:
            title = "Housing Market Dashboard"
        
        if description is None:
            description = "Generated by Fairfield County Housing Market Analysis Application"
        
        # Set default export directory
        if export_dir is None:
            export_dir = os.path.join('exports', 'dashboard')
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Create dashboard HTML file        
        html_file = os.path.join(export_dir, f"{title}.html")
        
        # Create dashboard HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>{title}</title>
            </head>
            <body>
                <h1>{title}</h1>
                <p>{description}</p>
                <div id="dashboard-content">
                    <div id="visualizations">
                        <h2>Visualizations</h2>
                    </div>
                </div>
            </body>
        </html>
        """
        
        # Write HTML content to file
        with open(html_file, 'w') as f:
            f.write(html_content)

        return html_file
    
    except Exception as e:
        logger.error(f"Error creating dashboard export: {str(e)}")
        return None

def export_data(df_dict, export_dir=None, formats=None, prefix=None):
    """
    Export multiple DataFrames to files.
    
    Args:
        df_dict (dict): Dictionary of DataFrames.
        export_dir (str, optional): Export directory. Defaults to None.
        formats (list, optional): List of export formats. Defaults to None.
        prefix (str, optional): Filename prefix. Defaults to None.
    
    Returns:
        dict: Dictionary of exported file paths.
    """
    try:
        if not df_dict:
            logger.error("No DataFrames provided for export")
            return {}
        
        # Set default export directory
        if export_dir is None:
            export_dir = os.path.join('exports', 'data')
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Set default formats
        if formats is None:
            formats = ['csv', 'excel']
        
        # Set default prefix
        if prefix is None:
            prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export each DataFrame
        exported_files = {}
        
        for name, df in df_dict.items():
            if df is None or df.empty:
                logger.warning(f"DataFrame '{name}' is empty or None, skipping export")
                continue
            
            # Create safe filename
            safe_name = name.replace(' ', '_').replace('/', '_').lower()
            
            # Export in each format
            for fmt in formats:
                if fmt.lower() == 'csv':
                    # Export as CSV
                    filename = f"{prefix}_{safe_name}.csv"
                    filepath = os.path.join(export_dir, filename)
                    
                    try:
                        df.to_csv(filepath, index=False)
                        exported_files[f"{name}_csv"] = filepath
                        logger.info(f"Exported {name} to {filepath}")
                    except Exception as e:
                        logger.error(f"Error exporting {name} to CSV: {str(e)}")
                
                elif fmt.lower() == 'excel':
                    # Export as Excel
                    filename = f"{prefix}_{safe_name}.xlsx"
                    filepath = os.path.join(export_dir, filename)
                    
                    try:
                        df.to_excel(filepath, sheet_name=name[:31], index=False)  # Excel sheet names limited to 31 chars
                        exported_files[f"{name}_excel"] = filepath
                        logger.info(f"Exported {name} to {filepath}")
                    except Exception as e:
                        logger.error(f"Error exporting {name} to Excel: {str(e)}")
                
                elif fmt.lower() == 'json':
                    # Export as JSON
                    filename = f"{prefix}_{safe_name}.json"
                    filepath = os.path.join(export_dir, filename)
                    
                    try:
                        df.to_json(filepath, orient='records', date_format='iso')
                        exported_files[f"{name}_json"] = filepath
                        logger.info(f"Exported {name} to {filepath}")
                    except Exception as e:
                        logger.error(f"Error exporting {name} to JSON: {str(e)}")
                
                else:
                    logger.warning(f"Unsupported export format: {fmt}")
        
        return exported_files
    
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return {}

def create_download_link(data, filename, mime_type):
    """
    Create a download link for data.
    
    Args:
        data (bytes or str): Data to download.
        filename (str): Download filename.
        mime_type (str): MIME type of the data.
    
    Returns:
        str: HTML download link.
    """ 
    try:
        if data is None:
            logger.error("No data provided for download link")
            return None
        
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode()
        
        # Create base64 encoded data
        b64 = base64.b64encode(data).decode()
        href = f'data:{mime_type};base64,{b64}'
        download_link = f'<a href="{href}" download="{filename}">Download {filename}</a>'
        
        return download_link
    
    except Exception as e:
        logger.error(f"Error creating download link: {str(e)}")     