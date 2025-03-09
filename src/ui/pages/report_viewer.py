import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import base64
from typing import Dict, Any, List

from src.ai_analysis.report_formatter import format_report, export_report

def display_report(report: Dict[str, Any], show_metadata: bool = True, allow_download: bool = True):
    """
    Display a report in the Streamlit UI.
    
    Args:
        report (Dict[str, Any]): Report data.
        show_metadata (bool, optional): Whether to show report metadata. Defaults to True.
        allow_download (bool, optional): Whether to allow report download. Defaults to True.
    """
    if not report or not isinstance(report, dict):
        st.error("Invalid report data")
        return
    
    # Extract content and metadata
    content = report.get('content', '')
    metadata = report.get('metadata', {})
    
    # Show report content
    st.markdown(content)
    
    # Show metadata if requested
    if show_metadata and metadata:
        with st.expander("Report Metadata"):
            # Format metadata as a table
            metadata_df = pd.DataFrame(
                [{"Key": k, "Value": str(v)} for k, v in metadata.items() if not isinstance(v, dict)]
            )
            st.table(metadata_df)
            
            # Show nested parameters if present
            if 'parameters' in metadata and isinstance(metadata['parameters'], dict):
                st.subheader("Report Parameters")
                params_df = pd.DataFrame(
                    [{"Parameter": k, "Value": str(v)} for k, v in metadata['parameters'].items()]
                )
                st.table(params_df)
    
    # Add download options if allowed
    if allow_download:
        show_download_options(report)

def show_download_options(report: Dict[str, Any]):
    """
    Show download options for a report.
    
    Args:
        report (Dict[str, Any]): Report data.
    """
    st.divider()
    st.subheader("Download Report")
    
    # Create download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download as HTML"):
            # Format as HTML
            html_content = format_report(report, format_type='html')
            
            # Create download link
            b64 = base64.b64encode(html_content.encode()).decode()
            href = f'data:text/html;base64,{b64}'
            
            # Create filename
            report_type = report.get('metadata', {}).get('report_type', 'report')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_{timestamp}.html"
            
            # Display download link
            st.markdown(
                f'<a href="{href}" download="{filename}">Click to download HTML</a>',
                unsafe_allow_html=True
            )
    
    with col2:
        if st.button("Download as DOCX"):
            # Create temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            
            # Generate filename
            report_type = report.get('metadata', {}).get('report_type', 'report')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_{timestamp}.docx"
            file_path = os.path.join("temp", filename)
            
            # Export report
            exported_path = export_report(report, file_path, format_type='docx')
            
            if exported_path:
                # Read the file
                with open(exported_path, 'rb') as f:
                    docx_bytes = f.read()
                
                # Create download button
                st.download_button(
                    label="Click to download DOCX",
                    data=docx_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    
    with col3:
        if st.button("Download as PDF"):
            # Format as PDF
            pdf_bytes = format_report(report, format_type='pdf')
            
            # Create filename
            report_type = report.get('metadata', {}).get('report_type', 'report')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_{timestamp}.pdf"
            
            # Create download button
            st.download_button(
                label="Click to download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf"
            )

def display_report_list(reports: List[Dict[str, Any]], selected_callback=None):
    """
    Display a list of reports with selectable items.
    
    Args:
        reports (List[Dict[str, Any]]): List of reports.
        selected_callback (callable, optional): Callback function when a report is selected. Defaults to None.
    """
    if not reports:
        st.info("No reports available")
        return
    
    # Create a container for the report list
    report_list_container = st.container()
    
    with report_list_container:
        st.subheader("Available Reports")
        
        # Create a card for each report
        for i, report in enumerate(reports):
            metadata = report.get('metadata', {})
            
            # Extract report information
            report_type = metadata.get('report_type', 'Report')
            generated_at = metadata.get('generated_at', 'Unknown date')
            model = f"{metadata.get('provider', '')} {metadata.get('model', '')}"
            
            # Create a unique key for this report
            report_key = f"report_{i}_{report_type}"
            
            # Create an expander for the report
            with st.expander(f"{report_type.capitalize()} Report - {generated_at}", expanded=False):
                # Show brief metadata
                st.write(f"**Generated:** {generated_at}")
                st.write(f"**Model:** {model}")
                
                # Show a preview (first 200 characters)
                content = report.get('content', '')
                preview = content[:200] + '...' if len(content) > 200 else content
                st.markdown(preview)
                
                # Add buttons for actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("View Full Report", key=f"view_{report_key}"):
                        if selected_callback:
                            selected_callback(report)
                        else:
                            # Default behavior: display the report in a new expander
                            with st.expander("Full Report", expanded=True):
                                display_report(report)
                
                with col2:
                    if st.button("Download", key=f"download_{report_key}"):
                        show_download_options(report)
                
                with col3:
                    if st.button("Edit", key=f"edit_{report_key}"):
                        st.session_state['editing_report'] = report
                        st.session_state['editing_report_index'] = i

def display_report_editor(report: Dict[str, Any], save_callback=None):
    """
    Display an editor for modifying a report.
    
    Args:
        report (Dict[str, Any]): Report to edit.
        save_callback (callable, optional): Callback function when the report is saved. Defaults to None.
    """
    if not report or not isinstance(report, dict):
        st.error("Invalid report data")
        return
    
    # Extract content and metadata
    content = report.get('content', '')
    metadata = report.get('metadata', {})
    
    st.subheader("Edit Report")
    
    # Create a text area for editing the content
    edited_content = st.text_area("Report Content", value=content, height=400)
    
    # Add metadata display (non-editable)
    with st.expander("Report Metadata (Read-only)"):
        metadata_df = pd.DataFrame(
            [{"Key": k, "Value": str(v)} for k, v in metadata.items() if not isinstance(v, dict)]
        )
        st.table(metadata_df)
    
    # Add save and cancel buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Changes"):
            # Update the report content
            updated_report = report.copy()
            updated_report['content'] = edited_content
            
            # Update metadata
            updated_report['metadata'] = metadata.copy()
            updated_report['metadata']['last_edited'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Call the save callback if provided
            if save_callback:
                save_callback(updated_report)
            else:
                st.success("Report updated")
                st.session_state['editing_report'] = None
                return updated_report
    
    with col2:
        if st.button("Cancel"):
            st.session_state['editing_report'] = None
            return None
    
    # Preview the edited report
    with st.expander("Preview Edited Report"):
        st.markdown(edited_content)
    
    return None

def manage_report_repository(repository_path: str = "reports"):
    """
    Manage a repository of saved reports.
    
    Args:
        repository_path (str, optional): Path to the reports repository. Defaults to "reports".
    
    Returns:
        List[Dict[str, Any]]: List of reports in the repository.
    """
    # Create repository directory if it doesn't exist
    os.makedirs(repository_path, exist_ok=True)
    
    # Load all reports from the repository
    reports = []
    
    for filename in os.listdir(repository_path):
        if filename.endswith('.json'):
            file_path = os.path.join(repository_path, filename)
            
            try:
                with open(file_path, 'r') as f:
                    report = json.load(f)
                
                if isinstance(report, dict) and 'content' in report:
                    reports.append(report)
            except Exception as e:
                st.error(f"Error loading report {filename}: {str(e)}")
    
    # Sort reports by generation date (newest first)
    reports.sort(
        key=lambda r: datetime.strptime(
            r.get('metadata', {}).get('generated_at', '2000-01-01 00:00:00'),
            "%Y-%m-%d %H:%M:%S"
        ), 
        reverse=True
    )
    
    return reports

def save_report_to_repository(report: Dict[str, Any], repository_path: str = "reports"):
    """
    Save a report to the repository.
    
    Args:
        report (Dict[str, Any]): Report to save.
        repository_path (str, optional): Path to the reports repository. Defaults to "reports".
    
    Returns:
        str: Path to the saved report file.
    """
    # Create repository directory if it doesn't exist
    os.makedirs(repository_path, exist_ok=True)
    
    # Generate filename
    metadata = report.get('metadata', {})
    report_type = metadata.get('report_type', 'report')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report_type}_{timestamp}.json"
    file_path = os.path.join(repository_path, filename)
    
    # Save report
    try:
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return file_path
    except Exception as e:
        st.error(f"Error saving report: {str(e)}")
        return None

def delete_report_from_repository(report_index: int, reports: List[Dict[str, Any]], repository_path: str = "reports"):
    """
    Delete a report from the repository.
    
    Args:
        report_index (int): Index of the report to delete.
        reports (List[Dict[str, Any]]): List of reports.
        repository_path (str, optional): Path to the reports repository. Defaults to "reports".
    
    Returns:
        bool: True if successful, False otherwise.
    """
    if report_index < 0 or report_index >= len(reports):
        st.error("Invalid report index")
        return False
    
    # Get report to delete
    report = reports[report_index]
    metadata = report.get('metadata', {})
    
    # Find the file in the repository
    for filename in os.listdir(repository_path):
        if not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(repository_path, filename)
        
        try:
            with open(file_path, 'r') as f:
                file_report = json.load(f)
            
            # Check if this is the report we want to delete
            file_metadata = file_report.get('metadata', {})
            
            if (file_metadata.get('report_type') == metadata.get('report_type') and
                file_metadata.get('generated_at') == metadata.get('generated_at')):
                # Found the file, delete it
                os.remove(file_path)
                return True
        except Exception:
            continue
    
    st.error("Report file not found in repository")
    return False
