"""
Logger Utility

This module provides utilities for setting up and configuring logging.
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional

def setup_logger(log_dir: Optional[str] = None, 
                log_level: str = "INFO", 
                file_logging: bool = True) -> None:
    """
    Setup application logging.
    
    Args:
        log_dir: Directory for log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        file_logging: Whether to log to file.
    """
    # Convert log level string to logging level
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_logging:
        # Set default log directory if not provided
        if log_dir is None:
            log_dir = "logs"
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"application_{timestamp}.log")
        
        # Create rotating file handler (max 10MB per file, keep 10 backup files)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024, 
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)

class LoggingContext:
    """
    Context manager for temporarily changing log level.
    """
    
    def __init__(self, logger, level=None, handler=None, close=True):
        """
        Initialize context manager.
        
        Args:
            logger: Logger to modify.
            level: Temporary log level.
            handler: Specific handler to modify.
            close: Whether to close the handler when exiting the context.
        """
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close
        
        # Save original level
        self.old_level = logger.level
        
        # Save handler level if specified
        if self.handler:
            self.old_handler_level = self.handler.level
    
    def __enter__(self):
        """
        Enter context manager.
        """
        if self.level is not None:
            self.logger.setLevel(self.level)
        
        if self.handler is not None and self.level is not None:
            self.handler.setLevel(self.level)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager.
        """
        # Restore original level
        self.logger.setLevel(self.old_level)
        
        # Restore handler level if specified
        if self.handler:
            self.handler.setLevel(self.old_handler_level)
            
            # Close handler if requested
            if self.close:
                self.handler.close()
                self.logger.removeHandler(self.handler)

def disable_external_loggers():
    """
    Disable loggers for external libraries to reduce noise.
    """
    # List of external libraries to silence
    external_loggers = [
        'urllib3',
        'requests',
        'matplotlib',
        'PIL',
        'plotly',
        'streamlit'
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def log_execution_time(logger, function_name=None):
    """
    Decorator to log execution time of a function.
    
    Args:
        logger: Logger to use.
        function_name: Name of the function (defaults to function.__name__).
        
    Returns:
        Decorated function.
    """
    def decorator(function):
        import functools
        import time
        
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            name = function_name or function.__name__
            start_time = time.time()
            
            logger.debug(f"Starting {name}")
            
            try:
                result = function(*args, **kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.debug(f"Finished {name} in {execution_time:.4f} seconds")
                
                return result
            
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.error(f"Error in {name} after {execution_time:.4f} seconds: {str(e)}")
                raise
        
        return wrapper
    
    return decorator
