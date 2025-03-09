"""
Caching Utilities

This module provides utilities for caching API responses and processed data.
"""

import os
import json
import pickle
import hashlib
import time
import logging
from functools import wraps
from typing import Any, Dict, Callable, Optional, Tuple, Union
import pandas as pd
from datetime import datetime, timedelta

# Configure logger
logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages caching of data to disk or memory.
    """
    
    def __init__(self, cache_dir: str = '.cache', ttl: int = 86400):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files.
            ttl: Time-to-live in seconds (default: 1 day).
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.memory_cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key.
            
        Returns:
            (hit, value) tuple where:
            - hit: Boolean indicating whether the key was found in the cache.
            - value: Cached value if hit is True, None otherwise.
        """
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            # Check if the entry has expired
            if entry['expires'] > time.time():
                logger.debug(f"Memory cache hit for key: {key}")
                return True, entry['value']
            
            # Remove expired entry
            del self.memory_cache[key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.cache")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check if the entry has expired
                if entry['expires'] > time.time():
                    logger.debug(f"Disk cache hit for key: {key}")
                    
                    # Add to memory cache for faster access
                    self.memory_cache[key] = entry
                    
                    return True, entry['value']
                
                # Remove expired cache file
                os.remove(cache_file)
            
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {str(e)}")
        
        return False, None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds (overrides default TTL if provided).
        """
        # Use default TTL if not provided
        if ttl is None:
            ttl = self.ttl
        
        # Create cache entry
        entry = {
            'value': value,
            'expires': time.time() + ttl
        }
        
        # Store in memory cache
        self.memory_cache[key] = entry
        
        # Store in disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.cache")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
            
            logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
        
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {str(e)}")
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Boolean indicating whether the key was deleted.
        """
        deleted = False
        
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            deleted = True
        
        # Remove from disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.cache")
        
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                deleted = True
                logger.debug(f"Deleted cache entry for key: {key}")
            
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_file}: {str(e)}")
        
        return deleted
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Pattern to match cache keys (None to clear all).
            
        Returns:
            Number of entries cleared.
        """
        count = 0
        
        # Clear memory cache
        if pattern is None:
            count += len(self.memory_cache)
            self.memory_cache.clear()
        else:
            keys_to_delete = [k for k in self.memory_cache if pattern in k]
            count += len(keys_to_delete)
            
            for key in keys_to_delete:
                del self.memory_cache[key]
        
        # Clear disk cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                key = filename[:-6]  # Remove .cache extension
                
                if pattern is None or pattern in key:
                    cache_file = os.path.join(self.cache_dir, filename)
                    
                    try:
                        os.remove(cache_file)
                        count += 1
                    except Exception as e:
                        logger.error(f"Error deleting cache file {cache_file}: {str(e)}")
        
        logger.info(f"Cleared {count} cache entries" + (f" matching '{pattern}'" if pattern else ""))
        
        return count
    
    def cleanup(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed.
        """
        count = 0
        
        # Clean up memory cache
        current_time = time.time()
        keys_to_delete = [k for k, v in self.memory_cache.items() if v['expires'] <= current_time]
        
        for key in keys_to_delete:
            del self.memory_cache[key]
            count += 1
        
        # Clean up disk cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                cache_file = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(cache_file, 'rb') as f:
                        entry = pickle.load(f)
                    
                    if entry['expires'] <= current_time:
                        os.remove(cache_file)
                        count += 1
                
                except Exception as e:
                    logger.error(f"Error processing cache file {cache_file}: {str(e)}")
                    
                    # Delete corrupted cache file
                    try:
                        os.remove(cache_file)
                        count += 1
                    except:
                        pass
        
        logger.info(f"Removed {count} expired cache entries")
        
        return count

# Create a global cache manager
_cache_manager = CacheManager()

def cache_response(ttl: Optional[int] = None):
    """
    Decorator for caching function responses.
    
    Args:
        ttl: Time-to-live in seconds (overrides default TTL if provided).
        
    Returns:
        Decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique key based on function name and arguments
            key_parts = [func.__name__]
            
            # Add positional arguments to key
            for arg in args:
                if isinstance(arg, (str, int, float, bool, type(None))):
                    key_parts.append(str(arg))
                else:
                    # For complex objects, use their string representation
                    key_parts.append(str(type(arg)))
                    key_parts.append(str(hash(str(arg))))
            
            # Add keyword arguments to key
            for k, v in sorted(kwargs.items()):
                key_parts.append(k)
                
                if isinstance(v, (str, int, float, bool, type(None))):
                    key_parts.append(str(v))
                else:
                    # For complex objects, use their string representation
                    key_parts.append(str(type(v)))
                    key_parts.append(str(hash(str(v))))
            
            # Create a hash from the key parts
            key = hashlib.md5(''.join(key_parts).encode()).hexdigest()
            
            # Try to get the value from the cache
            hit, value = _cache_manager.get(key)
            
            if hit:
                return value
            
            # Call the function if cache miss
            result = func(*args, **kwargs)
            
            # Cache the result
            _cache_manager.set(key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator

def cache_dataframe(filename: str, ttl: Optional[int] = None):
    """
    Decorator for caching pandas DataFrame responses to parquet files.
    
    Args:
        filename: Base filename for cache file.
        ttl: Time-to-live in seconds (overrides default TTL if provided).
        
    Returns:
        Decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique key based on function name and arguments
            key_parts = [func.__name__]
            
            # Add positional arguments to key
            for arg in args:
                if isinstance(arg, (str, int, float, bool, type(None))):
                    key_parts.append(str(arg))
                else:
                    # For complex objects, use their string representation
                    key_parts.append(str(type(arg)))
                    key_parts.append(str(hash(str(arg))))
            
            # Add keyword arguments to key
            for k, v in sorted(kwargs.items()):
                key_parts.append(k)
                
                if isinstance(v, (str, int, float, bool, type(None))):
                    key_parts.append(str(v))
                else:
                    # For complex objects, use their string representation
                    key_parts.append(str(type(v)))
                    key_parts.append(str(hash(str(v))))
            
            # Create a hash from the key parts
            key = hashlib.md5(''.join(key_parts).encode()).hexdigest()
            
            # Create cache filepath
            cache_file = os.path.join(_cache_manager.cache_dir, f"{filename}_{key}.parquet")
            meta_file = os.path.join(_cache_manager.cache_dir, f"{filename}_{key}.meta")
            
            # Check if cache file exists and is not expired
            if os.path.exists(cache_file) and os.path.exists(meta_file):
                try:
                    # Read metadata
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check expiration
                    if metadata.get('expires', 0) > time.time():
                        # Read DataFrame from parquet
                        df = pd.read_parquet(cache_file)
                        
                        logger.debug(f"DataFrame cache hit for {filename}_{key}")
                        
                        return df
                except Exception as e:
                    logger.error(f"Error reading DataFrame cache: {str(e)}")
            
            # Call the function if cache miss
            result = func(*args, **kwargs)
            
            # Ensure the result is a DataFrame
            if not isinstance(result, pd.DataFrame):
                logger.warning(f"Function {func.__name__} did not return a DataFrame, not caching")
                return result
            
            # Cache the result
            try:
                # Calculate expiration time
                expires = time.time() + (ttl if ttl is not None else _cache_manager.ttl)
                
                # Write DataFrame to parquet
                result.to_parquet(cache_file, index=True)
                
                # Write metadata
                metadata = {
                    'created': time.time(),
                    'expires': expires,
                    'function': func.__name__,
                    'shape': result.shape
                }
                
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f)
                
                logger.debug(f"Cached DataFrame to {cache_file}")
            
            except Exception as e:
                logger.error(f"Error caching DataFrame: {str(e)}")
            
            return result
        
        return wrapper
    
    return decorator

def clear_cache(pattern: Optional[str] = None) -> int:
    """
    Clear cache entries.
    
    Args:
        pattern: Pattern to match cache keys (None to clear all).
        
    Returns:
        Number of entries cleared.
    """
    return _cache_manager.clear(pattern)

def cleanup_cache() -> int:
    """
    Remove expired cache entries.
    
    Returns:
        Number of entries removed.
    """
    return _cache_manager.cleanup()
