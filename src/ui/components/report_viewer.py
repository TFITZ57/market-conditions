"""
Report Viewer Components

This module provides UI components for viewing and managing reports.
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, Any, List, Optional, Union

def display_report(report_content: Dict[str, Any], format_type: str = "html"):
    """
    Display a report in the Streamlit UI.
    
    Args:
        report_content: Dictionary containing report content
        format_type: Format to display ('html', 'markdown', 'text')
    """
    # Extract components from report structure
    title = report_content.get('title', 'Housing Market Analysis Report')
    summary = report_content.get('summary', 'No summary provided.')
    sections = report_content.get('sections', [])
    recommendations = report_content.get('recommendations', [])
    metadata = report_content.get('metadata', {})
    
    # Display the report
    st.title(title)
    
    # Display summary
    st.header("Executive Summary")
    st.write(summary)
    
    # Display sections
    for section in sections:
        section_title = section.get('title', 'Section')
        section_content = section.get('content', 'No content provided.')
        
        st.header(section_title)
        st.write(section_content)
    
    # Display recommendations if present
    if recommendations:
        st.header("Recommendations")
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"{i}. {recommendation}")
    
    # Display metadata in expander
    with st.expander("Report Information"):
        for key, value in metadata.items():
            st.write(f"**{key}:** {value}")

def display_report_list(reports: List[Dict[str, Any]], on_select=None):
    """
    Display a list of reports with selection capabilities.
    
    Args:
        reports: List of report dictionaries
        on_select: Callback function when a report is selected
    """
    if not reports:
        st.info("No reports available. Generate a new report to get started.")
        return None
    
    # Display reports as a table
    report_data = []
    for i, report in enumerate(reports):
        report_data.append({
            "ID": i,
            "Title": report.get('title', f"Report {i+1}"),
            "Type": report.get('metadata', {}).get('report_type', 'Unknown'),
            "Created": report.get('metadata', {}).get('generated_at', 'Unknown'),
        })
    
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df)
    
    # Create selection widget
    selected_id = st.selectbox(
        "Select a report to view:",
        options=report_df['ID'].tolist(),
        format_func=lambda x: report_df.loc[report_df['ID'] == x, 'Title'].iloc[0]
    )
    
    # Call callback if provided
    if on_select is not None and selected_id is not None:
        on_select(reports[selected_id])
    
    return selected_id

def display_report_editor(report_content: Dict[str, Any], on_save=None):
    """
    Display an editor for modifying reports.
    
    Args:
        report_content: Dictionary containing report content
        on_save: Callback function when report is saved
    """
    # Extract components from report structure
    title = report_content.get('title', 'Housing Market Analysis Report')
    summary = report_content.get('summary', 'No summary provided.')
    sections = report_content.get('sections', [])
    recommendations = report_content.get('recommendations', [])
    metadata = report_content.get('metadata', {})
    
    # Create editable fields
    st.subheader("Edit Report")
    
    new_title = st.text_input("Report Title", value=title)
    
    new_summary = st.text_area("Executive Summary", value=summary, height=200)
    
    # Edit sections
    st.subheader("Sections")
    new_sections = []
    
    for i, section in enumerate(sections):
        st.markdown(f"##### Section {i+1}")
        section_title = section.get('title', 'Section')
        section_content = section.get('content', 'No content provided.')
        
        new_section_title = st.text_input(f"Section {i+1} Title", value=section_title, key=f"sect_title_{i}")
        new_section_content = st.text_area(f"Section {i+1} Content", value=section_content, height=150, key=f"sect_content_{i}")
        
        new_sections.append({
            'title': new_section_title,
            'content': new_section_content
        })
        
        st.markdown("---")
    
    # Edit recommendations
    st.subheader("Recommendations")
    new_recommendations = []
    
    for i, recommendation in enumerate(recommendations):
        new_recommendation = st.text_area(f"Recommendation {i+1}", value=recommendation, height=100, key=f"rec_{i}")
        new_recommendations.append(new_recommendation)
    
    # Create updated report content
    updated_report = {
        'title': new_title,
        'summary': new_summary,
        'sections': new_sections,
        'recommendations': new_recommendations,
        'metadata': metadata  # Keep original metadata
    }
    
    # Save button
    if st.button("Save Changes"):
        if on_save:
            on_save(updated_report)
        else:
            st.success("Report updated. Note: Changes are not saved because no save handler was provided.")
        
        return updated_report
    
    return None

def manage_report_repository():
    """
    Create or get a repository of reports from session state.
    
    Returns:
        List of report dictionaries
    """
    if 'reports' not in st.session_state:
        st.session_state['reports'] = []
    
    return st.session_state['reports']

def save_report_to_repository(report: Dict[str, Any]):
    """
    Save a report to the repository.
    
    Args:
        report: Report dictionary to save
    """
    if 'reports' not in st.session_state:
        st.session_state['reports'] = []
    
    # If report has an ID, update it, otherwise add it
    if 'id' in report.get('metadata', {}):
        report_id = report['metadata']['id']
        for i, existing_report in enumerate(st.session_state['reports']):
            if existing_report.get('metadata', {}).get('id') == report_id:
                st.session_state['reports'][i] = report
                return
    
    # Add new report
    if 'metadata' not in report:
        report['metadata'] = {}
    
    # Generate ID if needed
    if 'id' not in report['metadata']:
        import uuid
        report['metadata']['id'] = str(uuid.uuid4())
    
    # Add timestamp if needed
    if 'generated_at' not in report['metadata']:
        from datetime import datetime
        report['metadata']['generated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    st.session_state['reports'].append(report)

def delete_report_from_repository(report_id: str):
    """
    Delete a report from the repository.
    
    Args:
        report_id: ID of the report to delete
    """
    if 'reports' not in st.session_state:
        return
    
    st.session_state['reports'] = [
        report for report in st.session_state['reports'] 
        if report.get('metadata', {}).get('id') != report_id
    ] 