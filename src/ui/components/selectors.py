"""
Selector Components

This module provides selection components for the Streamlit UI:
- MetricSelector: Component for selecting housing market metrics
- ReportTypeSelector: Component for selecting report types
- LLMProviderSelector: Component for selecting LLM providers
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from ...utils.logger import get_logger

logger = get_logger(__name__)

class MetricSelector:
    """Component for selecting housing market metrics."""
    
    def __init__(self, 
                available_metrics: Optional[List[str]] = None,
                default_metrics: Optional[List[str]] = None,
                allow_multi: bool = True,
                max_selections: Optional[int] = None,
                group_by_category: bool = True,
                key_prefix: str = "metric"):
        """
        Initialize the metric selector.
        
        Args:
            available_metrics: List of available metrics
            default_metrics: List of default selected metrics
            allow_multi: Whether to allow multiple selection
            max_selections: Maximum number of selections allowed
            group_by_category: Whether to group metrics by category
            key_prefix: Prefix for Streamlit widget keys
        """
        self.allow_multi = allow_multi
        self.max_selections = max_selections
        self.group_by_category = group_by_category
        self.key_prefix = key_prefix
        
        # Default housing market metrics
        default_available_metrics = [
            "Listing Inventory", 
            "Months of Supply", 
            "New Listings", 
            "Housing Starts", 
            "Sales Volume",
            "Days on Market", 
            "Absorption Rate", 
            "Pending Home Sales", 
            "Median Sale Price",
            "Average Sale Price", 
            "LP/SP Ratio",
            "Home Price-to-Income Ratio",
            "Mortgage Rates", 
            "Housing Affordability Index", 
            "Local Job Growth",
            "Employment Trends", 
            "Vacancy Rates", 
            "Seller Concessions"
        ]
        
        self.available_metrics = available_metrics or default_available_metrics
        
        # Default selected metrics
        default_selected = [
            "Median Sale Price",
            "Months of Supply",
            "Days on Market",
            "LP/SP Ratio",
            "Home Price-to-Income Ratio"
        ]
        
        # Filter default selections to only include available metrics
        filtered_defaults = [m for m in (default_metrics or default_selected) if m in self.available_metrics]
        
        # Limit to max_selections if specified
        if max_selections and len(filtered_defaults) > max_selections:
            filtered_defaults = filtered_defaults[:max_selections]
            
        self.default_metrics = filtered_defaults
        
        # Group metrics by category if requested
        if group_by_category:
            self.metric_categories = {
                "Price": [
                    "Median Sale Price",
                    "Average Sale Price",
                    "LP/SP Ratio",
                    "Home Price-to-Income Ratio",
                    "Seller Concessions"
                ],
                "Inventory": [
                    "Listing Inventory",
                    "Months of Supply",
                    "New Listings",
                    "Housing Starts",
                    "Vacancy Rates"
                ],
                "Market Activity": [
                    "Sales Volume",
                    "Days on Market",
                    "Absorption Rate",
                    "Pending Home Sales"
                ],
                "Economic": [
                    "Mortgage Rates",
                    "Housing Affordability Index",
                    "Local Job Growth",
                    "Employment Trends"
                ]
            }
        else:
            self.metric_categories = {}
    
    def render(self) -> Union[str, List[str]]:
        """
        Render the metric selector.
        
        Returns:
            Selected metric(s)
        """
        st.subheader("Market Metrics")
        
        if self.group_by_category and self.metric_categories:
            return self._render_grouped()
        else:
            return self._render_flat()
    
    def _render_flat(self) -> Union[str, List[str]]:
        """
        Render a flat (non-grouped) metric selector.
        
        Returns:
            Selected metric(s)
        """
        if self.allow_multi:
            selected_metrics = st.multiselect(
                "Select Metrics",
                options=self.available_metrics,
                default=self.default_metrics,
                key=f"{self.key_prefix}_multi",
                help="Select the housing market metrics to analyze"
            )
            
            # Enforce maximum selections if specified
            if self.max_selections and len(selected_metrics) > self.max_selections:
                st.warning(f"Maximum of {self.max_selections} metrics allowed. Only the first {self.max_selections} will be used.")
                selected_metrics = selected_metrics[:self.max_selections]
                
            return selected_metrics
        else:
            default_idx = 0
            if self.default_metrics and len(self.default_metrics) > 0:
                try:
                    default_idx = self.available_metrics.index(self.default_metrics[0])
                except ValueError:
                    default_idx = 0
            
            selected_metric = st.selectbox(
                "Select Metric",
                options=self.available_metrics,
                index=default_idx,
                key=f"{self.key_prefix}_single",
                help="Select a housing market metric to analyze"
            )
            
            return selected_metric
    
    def _render_grouped(self) -> List[str]:
        """
        Render a grouped metric selector with expanders for each category.
        
        Returns:
            List of selected metrics
        """
        selected_metrics = []
        
        for category, metrics in self.metric_categories.items():
            # Filter to only include available metrics
            category_metrics = [m for m in metrics if m in self.available_metrics]
            
            if not category_metrics:
                continue
                
            # Determine if expander should be expanded by default
            default_expanded = any(m in self.default_metrics for m in category_metrics)
            
            with st.expander(category, expanded=default_expanded):
                for metric in category_metrics:
                    default_checked = metric in self.default_metrics
                    
                    if st.checkbox(
                        metric,
                        value=default_checked,
                        key=f"{self.key_prefix}_{category}_{metric.replace(' ', '_')}"
                    ):
                        selected_metrics.append(metric)
        
        # Enforce maximum selections if specified
        if self.max_selections and len(selected_metrics) > self.max_selections:
            st.warning(f"Maximum of {self.max_selections} metrics allowed. Only the first {self.max_selections} will be used.")
            selected_metrics = selected_metrics[:self.max_selections]
            
        # Display selected metrics count
        if selected_metrics:
            st.caption(f"Selected {len(selected_metrics)} metrics")
        else:
            st.warning("Please select at least one metric")
            
        return selected_metrics


class ReportTypeSelector:
    """Component for selecting report types."""
    
    def __init__(self, 
                report_types: Optional[Dict[str, str]] = None,
                default_type: Optional[str] = None,
                key_prefix: str = "report_type"):
        """
        Initialize the report type selector.
        
        Args:
            report_types: Dictionary mapping report type keys to descriptions
            default_type: Default selected report type
            key_prefix: Prefix for Streamlit widget keys
        """
        self.key_prefix = key_prefix
        
        # Default report types
        default_report_types = {
            "general_analysis": "General Analysis - Overall market health and trends",
            "investment_analysis": "Investment Analysis - Opportunities and risks for investors",
            "market_forecast": "Market Forecast - Projected market movements with confidence levels",
            "comparative_analysis": "Comparative Analysis - Cross-location or cross-time period analysis",
            "selling_recommendations": "Selling Recommendations - Optimal timing and pricing strategies",
            "buying_recommendations": "Buying Recommendations - Value opportunities and negotiation guidance"
        }
        
        self.report_types = report_types or default_report_types
        self.default_type = default_type or "general_analysis"
        
        # Ensure default type is valid
        if self.default_type not in self.report_types:
            self.default_type = list(self.report_types.keys())[0]
    
    def render(self) -> Tuple[str, str]:
        """
        Render the report type selector.
        
        Returns:
            Tuple of (report_type_key, report_type_description)
        """
        st.subheader("Report Type")
        
        report_type_options = list(self.report_types.keys())
        report_type_descriptions = list(self.report_types.values())
        
        # Find default index
        try:
            default_idx = report_type_options.index(self.default_type)
        except ValueError:
            default_idx = 0
        
        # Radio selection with descriptions
        selected_idx = st.radio(
            "Select Report Type",
            options=range(len(report_type_options)),
            format_func=lambda i: report_type_descriptions[i],
            index=default_idx,
            key=f"{self.key_prefix}_radio"
        )
        
        selected_type = report_type_options[selected_idx]
        selected_description = report_type_descriptions[selected_idx]
        
        return selected_type, selected_description


class LLMProviderSelector:
    """Component for selecting LLM providers and models."""
    
    def __init__(self,
                providers: Optional[Dict[str, List[str]]] = None,
                default_provider: Optional[str] = None,
                default_model: Optional[str] = None,
                show_tone_selector: bool = True,
                key_prefix: str = "llm"):
        """
        Initialize the LLM provider selector.
        
        Args:
            providers: Dictionary mapping provider names to lists of model names
            default_provider: Default selected provider
            default_model: Default selected model
            show_tone_selector: Whether to show the tone selection component
            key_prefix: Prefix for Streamlit widget keys
        """
        self.key_prefix = key_prefix
        self.show_tone_selector = show_tone_selector
        
        # Default providers and models
        default_providers = {
            "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "Anthropic": ["claude-3.5-sonnet-20241022", "claude-3.7-sonnet"]
        }
        
        self.providers = providers or default_providers
        self.default_provider = default_provider or "OpenAI"
        
        # Ensure default provider is valid
        if self.default_provider not in self.providers:
            self.default_provider = list(self.providers.keys())[0]
            
        # Get models for default provider
        models = self.providers.get(self.default_provider, [])
        
        # Set default model
        if default_model and default_model in models:
            self.default_model = default_model
        elif models:
            self.default_model = models[0]
        else:
            self.default_model = None
            
        # Tone options
        self.tones = [
            "Analytical", 
            "Informative", 
            "Casual", 
            "Skeptical", 
            "Academic",
            "Professional",
            "Concise"
        ]
    
    def render(self) -> Dict[str, str]:
        """
        Render the LLM provider selector.
        
        Returns:
            Dictionary with 'provider', 'model', and optionally 'tone' keys
        """
        st.subheader("LLM Settings")
        
        cols = st.columns(2)
        
        with cols[0]:
            # Provider selection
            provider_options = list(self.providers.keys())
            
            try:
                default_provider_idx = provider_options.index(self.default_provider)
            except ValueError:
                default_provider_idx = 0
                
            selected_provider = st.selectbox(
                "Provider",
                options=provider_options,
                index=default_provider_idx,
                key=f"{self.key_prefix}_provider"
            )
        
        with cols[1]:
            # Model selection based on provider
            model_options = self.providers.get(selected_provider, [])
            
            if not model_options:
                st.warning(f"No models available for {selected_provider}")
                selected_model = None
            else:
                # Set default model for selected provider
                if selected_provider == self.default_provider and self.default_model in model_options:
                    default_model_idx = model_options.index(self.default_model)
                else:
                    default_model_idx = 0
                    
                selected_model = st.selectbox(
                    "Model",
                    options=model_options,
                    index=default_model_idx,
                    key=f"{self.key_prefix}_model"
                )
        
        # Tone selection
        selected_tone = None
        if self.show_tone_selector:
            selected_tone = st.selectbox(
                "Tone",
                options=self.tones,
                index=0,  # Default to Analytical
                key=f"{self.key_prefix}_tone",
                help="Select the writing tone for generated reports"
            )
        
        # Construct result
        result = {
            "provider": selected_provider,
            "model": selected_model
        }
        
        if selected_tone:
            result["tone"] = selected_tone
            
        return result 