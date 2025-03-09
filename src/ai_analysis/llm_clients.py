"""
LLM Clients

This module provides client interfaces for various LLM providers.
"""

import logging
import os
from typing import Dict, Any, Optional, Iterator, Union

# Import actual clients
from src.ai_analysis.openai_client import OpenAIClient
from src.ai_analysis.anthropic_client import AnthropicClient

# Configure logger
logger = logging.getLogger(__name__)

def get_client(provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None) -> Union[OpenAIClient, AnthropicClient]:
    """
    Get an LLM client for the specified provider and model.
    
    Args:
        provider: LLM provider ('openai', 'anthropic', etc.)
        api_key: API key for the provider (defaults to environment variable)
        model: Specific model to use
        
    Returns:
        LLM client instance
    """
    if provider.lower() == "openai":
        model = model or "gpt-4o"
        return OpenAIClient(api_key=api_key, model=model)
    elif provider.lower() == "anthropic":
        model = model or "claude-3-5-sonnet"
        return AnthropicClient(api_key=api_key, model=model)
    else:
        logger.warning(f"Unknown provider: {provider}. Defaulting to OpenAI.")
        model = model or "gpt-4o"
        return OpenAIClient(api_key=api_key, model=model)

def get_llm_client(provider: str = "openai", model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get an LLM client interface for the specified provider and model.
    
    Args:
        provider: LLM provider ('openai', 'anthropic', etc.)
        model: Specific model to use
        
    Returns:
        Client interface object
    """
    if provider.lower() == "openai":
        return get_openai_client(model)
    elif provider.lower() == "anthropic":
        return get_anthropic_client(model)
    else:
        logger.warning(f"Unknown provider: {provider}. Defaulting to OpenAI.")
        return get_openai_client(model)

def get_openai_client(model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get an OpenAI client interface.
    
    Args:
        model: Model name (e.g., 'gpt-4o', 'gpt-4o-mini')
        
    Returns:
        OpenAI client interface
    """
    # Set default model if not specified
    if model is None:
        model = "gpt-4o"
    
    # Create a real OpenAI client
    client = OpenAIClient(model=model)
    
    def generate(prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, str]:
        response = client.generate_completion(prompt, max_tokens=max_tokens, temperature=temperature)
        return {"content": response["text"]}
    
    def stream(prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Iterator[Dict[str, str]]:
        for chunk in client.stream_completion(prompt, max_tokens=max_tokens, temperature=temperature):
            yield {"content": chunk["text"]}
    
    return {
        "provider": "openai",
        "model": model,
        "generate": generate,
        "stream": stream
    }

def get_anthropic_client(model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get an Anthropic client interface.
    
    Args:
        model: Model name (e.g., 'claude-3-5-sonnet', 'claude-3-opus')
        
    Returns:
        Anthropic client interface
    """
    # Set default model if not specified
    if model is None:
        model = "claude-3-5-sonnet"
    
    # Create a real Anthropic client
    client = AnthropicClient(model=model)
    
    def generate(prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, str]:
        response = client.generate_completion(prompt, max_tokens=max_tokens, temperature=temperature)
        return {"content": response["text"]}
    
    def stream(prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Iterator[Dict[str, str]]:
        for chunk in client.stream_completion(prompt, max_tokens=max_tokens, temperature=temperature):
            yield {"content": chunk["text"]}
    
    return {
        "provider": "anthropic",
        "model": model,
        "generate": generate,
        "stream": stream
    }

def generate_response(client: Dict[str, Any], prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """
    Generate a response from an LLM.
    
    Args:
        client: LLM client interface
        prompt: Input prompt
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated response text
    """
    try:
        response = client["generate"](prompt, temperature=temperature, max_tokens=max_tokens)
        return response["content"]
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}" 