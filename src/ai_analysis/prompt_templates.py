"""
Prompt Templates

This module provides template generation functions for LLM prompts.
"""

from typing import Dict, Any, List
import json

def create_full_prompt(report_type: str, analysis_data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
    """
    Create a full prompt for LLM report generation based on the report type and data.
    
    Args:
        report_type: Type of report to generate
        analysis_data: Data for analysis
        parameters: Additional parameters for prompt customization
        
    Returns:
        Formatted prompt string
    """
    # Get report type specific template
    if report_type == "General Analysis":
        template = GENERAL_ANALYSIS_TEMPLATE
    elif report_type == "Investment Analysis":
        template = INVESTMENT_ANALYSIS_TEMPLATE
    elif report_type == "Market Forecast":
        template = MARKET_FORECAST_TEMPLATE
    elif report_type == "Comparative Analysis":
        template = COMPARATIVE_ANALYSIS_TEMPLATE
    elif report_type == "Selling Recommendations":
        template = SELLING_RECOMMENDATIONS_TEMPLATE
    elif report_type == "Buying Recommendations":
        template = BUYING_RECOMMENDATIONS_TEMPLATE
    else:
        template = GENERAL_ANALYSIS_TEMPLATE
    
    # Get tone modifier
    tone = parameters.get('tone', 'Informative')
    tone_instructions = TONE_INSTRUCTIONS.get(tone.lower(), TONE_INSTRUCTIONS['informative'])
    
    # Get length modifier
    length = parameters.get('length', 'Standard')
    length_instructions = LENGTH_INSTRUCTIONS.get(length.lower(), LENGTH_INSTRUCTIONS['standard'])
    
    # Format the data for inclusion in the prompt
    data_summary = format_data_for_prompt(analysis_data)
    
    # Build the full prompt
    prompt = f"""
{template}

{tone_instructions}

{length_instructions}

The following data is available for your analysis:

{data_summary}

Please provide your analysis based on this data.
"""
    
    return prompt

def create_json_format_instructions(schema: Dict[str, Any]) -> str:
    """
    Create format instructions for structured JSON output.
    
    Args:
        schema: JSON schema defining the expected output format
        
    Returns:
        Format instructions string
    """
    schema_str = json.dumps(schema, indent=2)
    
    instructions = f"""
Please format your response as a JSON object that conforms to the following schema:

{schema_str}

Ensure that your response can be parsed as valid JSON.
"""
    
    return instructions

def format_data_for_prompt(analysis_data: Dict[str, Any]) -> str:
    """
    Format analysis data for inclusion in prompts.
    
    Args:
        analysis_data: Dictionary containing analysis data
        
    Returns:
        Formatted string representation of the data
    """
    sections = []
    
    # Format summary statistics
    if 'summary_statistics' in analysis_data:
        stats = analysis_data['summary_statistics']
        stats_section = "SUMMARY STATISTICS:\n"
        
        for metric, values in stats.items():
            stats_section += f"- {metric}:\n"
            for stat_name, stat_value in values.items():
                stats_section += f"  - {stat_name}: {stat_value}\n"
        
        sections.append(stats_section)
    
    # Format key trends
    if 'key_trends' in analysis_data:
        trends = analysis_data['key_trends']
        trends_section = "KEY TRENDS:\n"
        
        for metric, trend_data in trends.items():
            trends_section += f"- {metric}:\n"
            for trend_name, trend_value in trend_data.items():
                trends_section += f"  - {trend_name}: {trend_value}\n"
        
        sections.append(trends_section)
    
    return "\n\n".join(sections)

# Template constants
GENERAL_ANALYSIS_TEMPLATE = """
You are a real estate market analysis expert. Create a comprehensive analysis of the Fairfield County housing market using the provided data. Your analysis should include current market conditions, trends, and their implications for different stakeholders.
"""

INVESTMENT_ANALYSIS_TEMPLATE = """
You are a real estate investment advisor. Analyze the provided Fairfield County housing market data to identify investment opportunities and risks. Your analysis should include ROI considerations, market timing advice, and potential growth areas.
"""

MARKET_FORECAST_TEMPLATE = """
You are a real estate market forecaster. Based on the provided Fairfield County housing market data, create a forecast of future market conditions. Your forecast should include price trends, inventory projections, and factors that might influence the market.
"""

COMPARATIVE_ANALYSIS_TEMPLATE = """
You are a comparative market analyst. Analyze the provided data to compare different towns within Fairfield County. Your analysis should highlight differences in market conditions, price points, and growth potential.
"""

SELLING_RECOMMENDATIONS_TEMPLATE = """
You are a real estate selling consultant. Based on the provided Fairfield County market data, provide strategic recommendations for sellers. Your recommendations should include timing, pricing strategies, and market positioning.
"""

BUYING_RECOMMENDATIONS_TEMPLATE = """
You are a real estate buying consultant. Based on the provided Fairfield County market data, provide strategic recommendations for buyers. Your recommendations should include areas of opportunity, timing considerations, and negotiation insights.
"""

# Tone instructions
TONE_INSTRUCTIONS = {
    "analytical": "Maintain a highly analytical tone with emphasis on data-driven insights and logical reasoning. Use precise language and focus on objective interpretations of the data.",
    "informative": "Use a balanced, informative tone that explains market conditions clearly. Provide context for data points and make information accessible while maintaining professional authority.",
    "casual": "Adopt a conversational, approachable tone that makes complex market data understandable to general audiences. Use simpler language and relatable examples.",
    "skeptical": "Maintain a cautious, questioning tone that critically examines market trends. Highlight potential issues with data interpretation and challenge common assumptions.",
    "academic": "Use a scholarly tone with formal language and structured argumentation. Reference methodological considerations and provide nuanced analysis of market dynamics."
}

# Length instructions
LENGTH_INSTRUCTIONS = {
    "brief": "Provide a concise analysis focusing only on the most critical insights. Limit your response to approximately 250-300 words.",
    "standard": "Provide a comprehensive analysis with balanced depth. Cover all relevant aspects while maintaining focus. Aim for approximately 500-700 words.",
    "comprehensive": "Deliver an in-depth analysis that thoroughly explores all aspects of the data. Include detailed explanations, nuanced interpretations, and broader context. Your response may be 1000+ words."
} 