"""
Filter Components

This module provides filter components for the Streamlit UI:
- DateRangeFilter: Date range selection with quarter granularity
- LocationFilter: Hierarchical location selection (county -> town)
- PropertyTypeFilter: Property type selection component
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from ...utils.logger import get_logger

logger = get_logger(__name__)

class DateRangeFilter:
    """Date range selection component with quarter granularity."""
    
    def __init__(self, 
                min_date: str = "2015-01-01", 
                max_date: Optional[str] = None,
                default_start_date: Optional[str] = None,
                default_end_date: Optional[str] = None,
                key_prefix: str = "date_range"):
        """
        Initialize the date range filter.
        
        Args:
            min_date: Minimum selectable date in YYYY-MM-DD format
            max_date: Maximum selectable date in YYYY-MM-DD format (defaults to current date)
            default_start_date: Default start date in YYYY-MM-DD format
            default_end_date: Default end date in YYYY-MM-DD format
            key_prefix: Prefix for Streamlit widget keys
        """
        self.min_date = pd.to_datetime(min_date)
        
        if max_date:
            self.max_date = pd.to_datetime(max_date)
        else:
            self.max_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        
        # Set defaults (1 year ago to now if not specified)
        if default_start_date:
            self.default_start_date = pd.to_datetime(default_start_date)
        else:
            self.default_start_date = self.max_date - timedelta(days=365)
            
        if default_end_date:
            self.default_end_date = pd.to_datetime(default_end_date)
        else:
            self.default_end_date = self.max_date
        
        # Ensure defaults are within bounds
        self.default_start_date = max(self.min_date, self.default_start_date)
        self.default_end_date = min(self.max_date, self.default_end_date)
        
        self.key_prefix = key_prefix
        
        # Generate quarter options
        self.quarters = self._generate_quarter_options()
    
    def _generate_quarter_options(self) -> List[str]:
        """
        Generate list of quarter options from min_date to max_date.
        
        Returns:
            List of quarter options in 'Q1 YYYY' format
        """
        # Round dates to start of quarter
        start_year = self.min_date.year
        start_quarter = (self.min_date.month - 1) // 3 + 1
        
        end_year = self.max_date.year
        end_quarter = (self.max_date.month - 1) // 3 + 1
        
        quarters = []
        
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                # Skip quarters before start or after end
                if (year == start_year and quarter < start_quarter) or \
                   (year == end_year and quarter > end_quarter):
                    continue
                
                quarters.append(f"Q{quarter} {year}")
        
        return quarters
    
    def _quarter_to_date(self, quarter_str: str) -> datetime:
        """
        Convert quarter string to datetime.
        
        Args:
            quarter_str: Quarter string in 'Q1 YYYY' format
            
        Returns:
            Datetime object for the start of the quarter
        """
        q = int(quarter_str[1])
        year = int(quarter_str[3:])
        month = (q - 1) * 3 + 1
        
        return datetime(year, month, 1)
    
    def render(self) -> Tuple[datetime, datetime]:
        """
        Render the date range filter.
        
        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        st.subheader("Date Range")
        
        # Find default quarter indices
        default_start_quarter = f"Q{(self.default_start_date.month - 1) // 3 + 1} {self.default_start_date.year}"
        default_end_quarter = f"Q{(self.default_end_date.month - 1) // 3 + 1} {self.default_end_date.year}"
        
        try:
            default_start_idx = self.quarters.index(default_start_quarter)
        except ValueError:
            default_start_idx = 0
            
        try:
            default_end_idx = self.quarters.index(default_end_quarter)
        except ValueError:
            default_end_idx = len(self.quarters) - 1
        
        # Create quarter selection
        cols = st.columns(2)
        
        with cols[0]:
            start_quarter = st.selectbox(
                "Start Quarter",
                options=self.quarters,
                index=default_start_idx,
                key=f"{self.key_prefix}_start"
            )
            
        with cols[1]:
            # Filter end quarters to be >= start_quarter
            start_idx = self.quarters.index(start_quarter)
            valid_end_quarters = self.quarters[start_idx:]
            
            # Adjust default end index for the filtered list
            adjusted_end_idx = min(default_end_idx - start_idx, len(valid_end_quarters) - 1)
            adjusted_end_idx = max(0, adjusted_end_idx)
            
            end_quarter = st.selectbox(
                "End Quarter",
                options=valid_end_quarters,
                index=adjusted_end_idx,
                key=f"{self.key_prefix}_end"
            )
        
        # Convert to datetime
        start_date = self._quarter_to_date(start_quarter)
        end_date = self._quarter_to_date(end_quarter)
        
        # For end date, set to end of quarter
        if end_quarter.startswith("Q1"):
            end_date = end_date.replace(month=3, day=31)
        elif end_quarter.startswith("Q2"):
            end_date = end_date.replace(month=6, day=30)
        elif end_quarter.startswith("Q3"):
            end_date = end_date.replace(month=9, day=30)
        elif end_quarter.startswith("Q4"):
            end_date = end_date.replace(month=12, day=31)
        
        # Display selected range
        date_range_str = f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
        st.caption(f"Selected range: {date_range_str}")
        
        return start_date, end_date


class LocationFilter:
    """Hierarchical location selection component (county -> town)."""
    
    def __init__(self, 
                fairfield_towns: Optional[List[str]] = None,
                default_value: str = "Fairfield County",
                include_all_option: bool = True,
                key_prefix: str = "location"):
        """
        Initialize the location filter.
        
        Args:
            fairfield_towns: List of towns in Fairfield County
            default_value: Default selected location
            include_all_option: Whether to include an 'All Towns' option
            key_prefix: Prefix for Streamlit widget keys
        """
        self.default_value = default_value
        self.include_all_option = include_all_option
        self.key_prefix = key_prefix
        
        # Default list of towns in Fairfield County, CT
        default_towns = [
            "Bridgeport", "Danbury", "Darien", "Fairfield", "Greenwich",
            "New Canaan", "Norwalk", "Stamford", "Stratford", "Westport",
            "Weston", "Wilton", "Bethel", "Brookfield", "Easton",
            "Monroe", "Newtown", "Redding", "Ridgefield", "Shelton", "Trumbull"
        ]
        
        self.towns = fairfield_towns or default_towns
        self.towns.sort()  # Alphabetical order
        
        # Create location options
        self.location_options = ["Fairfield County"]
        
        if include_all_option:
            self.location_options.append("All Towns")
            
        self.location_options.extend(self.towns)
    
    def render(self) -> str:
        """
        Render the location filter.
        
        Returns:
            Selected location (county, 'All Towns', or specific town)
        """
        st.subheader("Location")
        
        try:
            default_idx = self.location_options.index(self.default_value)
        except ValueError:
            default_idx = 0
        
        selected_location = st.selectbox(
            "Select Location",
            options=self.location_options,
            index=default_idx,
            key=f"{self.key_prefix}_location"
        )
        
        return selected_location
    
    def render_multi(self) -> List[str]:
        """
        Render a multi-select location filter.
        
        Returns:
            List of selected locations
        """
        st.subheader("Locations")
        
        selected_locations = st.multiselect(
            "Select Locations",
            options=self.location_options,
            default=["Fairfield County"],
            key=f"{self.key_prefix}_locations_multi"
        )
        
        return selected_locations


class PropertyTypeFilter:
    """Property type selection component."""
    
    def __init__(self, 
                property_types: Optional[List[str]] = None,
                default_value: Optional[str] = None,
                allow_multi: bool = False,
                key_prefix: str = "property_type"):
        """
        Initialize the property type filter.
        
        Args:
            property_types: List of property types
            default_value: Default selected property type
            allow_multi: Whether to allow multiple selection
            key_prefix: Prefix for Streamlit widget keys
        """
        # Default property types
        default_types = [
            "Single-family", 
            "Multi-unit", 
            "Condo/Townhouse", 
            "All Residential"
        ]
        
        self.property_types = property_types or default_types
        self.default_value = default_value or "All Residential"
        self.allow_multi = allow_multi
        self.key_prefix = key_prefix
    
    def render(self) -> Union[str, List[str]]:
        """
        Render the property type filter.
        
        Returns:
            Selected property type(s)
        """
        st.subheader("Property Type")
        
        if self.allow_multi:
            selected_types = st.multiselect(
                "Select Property Types",
                options=self.property_types,
                default=[self.default_value],
                key=f"{self.key_prefix}_types_multi"
            )
            
            return selected_types
        else:
            try:
                default_idx = self.property_types.index(self.default_value)
            except ValueError:
                default_idx = 0
                
            selected_type = st.selectbox(
                "Select Property Type",
                options=self.property_types,
                index=default_idx,
                key=f"{self.key_prefix}_type"
            )
            
            return selected_type 