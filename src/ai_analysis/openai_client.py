"""
OpenAI Client

This module provides a client interface for the OpenAI API.
"""

import os
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

# Configure logger
logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Client for OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY environment variable.
            model: Model name. Defaults to gpt-4o.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass api_key.")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_completion(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a completion from OpenAI.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Dict containing response text and usage information.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error generating completion from OpenAI: {str(e)}")
            raise
    
    def stream_completion(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7):
        """
        Stream a completion from OpenAI.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Yields:
            Dict containing chunk of response text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield {"text": chunk.choices[0].delta.content}
        except Exception as e:
            logger.error(f"Error streaming completion from OpenAI: {str(e)}")
            raise 