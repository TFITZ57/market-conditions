"""
Anthropic Client

This module provides a client interface for the Anthropic API.
"""

import os
import logging
from typing import Dict, Any, Optional
from anthropic import Anthropic

# Configure logger
logger = logging.getLogger(__name__)

class AnthropicClient:
    """
    Client for Anthropic API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet"):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY environment variable.
            model: Model name. Defaults to claude-3-5-sonnet.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key.")
        self.model = model
        self.client = Anthropic(api_key=self.api_key)
    
    def generate_completion(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a completion from Anthropic.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Dict containing response text and usage information.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response.content[0].text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error generating completion from Anthropic: {str(e)}")
            raise
    
    def stream_completion(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7):
        """
        Stream a completion from Anthropic.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Yields:
            Dict containing chunk of response text.
        """
        try:
            with self.client.messages.stream(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            ) as stream:
                for text in stream.text_stream:
                    yield {"text": text}
        except Exception as e:
            logger.error(f"Error streaming completion from Anthropic: {str(e)}")
            raise 