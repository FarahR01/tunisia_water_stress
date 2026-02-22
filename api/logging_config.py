"""Structured logging configuration for the API.

This module provides:
- JSON or console formatted logging
- Request correlation IDs
- Automatic request/response logging middleware
- Log context management
"""
import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


# Context variable for request-scoped data
request_context: ContextVar[dict[str, Any]] = ContextVar("request_context", default={})


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get context
        ctx = request_context.get()
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request context if available
        if ctx:
            log_entry["request_id"] = ctx.get("request_id")
            log_entry["client_ip"] = ctx.get("client_ip")
            log_entry["path"] = ctx.get("path")
            log_entry["method"] = ctx.get("method")
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        ctx = request_context.get()
        request_id = ctx.get("request_id", "-")[:8] if ctx.get("request_id") else "-"
        
        color = self.COLORS.get(record.levelname, "")
        
        # Build prefix
        prefix = f"{color}[{record.levelname}]{self.RESET}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"{timestamp} {prefix} [{request_id}] {record.name}: {record.getMessage()}"


def setup_logging(log_level: str = "INFO", log_format: str = "console") -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ('json' for structured, 'console' for development)
        
    Returns:
        Configured root logger
    """
    # Get root logger
    logger = logging.getLogger("api")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    
    # Set formatter based on format
    if log_format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())
    
    logger.addHandler(handler)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "api") -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging with correlation IDs.
    
    Adds:
    - Request ID generation and tracking
    - Request/response timing
    - Structured logging of HTTP transactions
    """
    
    def __init__(self, app, logger: logging.Logger | None = None):
        super().__init__(app)
        self.logger = logger or get_logger("api.http")
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with logging."""
        # Generate or get request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Get client IP (handle proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        client_ip = forwarded.split(",")[0].strip() if forwarded else (
            request.client.host if request.client else "unknown"
        )
        
        # Set context
        ctx = {
            "request_id": request_id,
            "client_ip": client_ip,
            "path": request.url.path,
            "method": request.method,
        }
        token = request_context.set(ctx)
        
        # Log request
        start_time = time.perf_counter()
        self.logger.info(
            f"Request: {request.method} {request.url.path}",
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log response
            self.logger.info(
                f"Response: {response.status_code} ({duration_ms:.2f}ms)"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.logger.error(
                f"Request failed: {type(e).__name__}: {str(e)} ({duration_ms:.2f}ms)"
            )
            raise
        finally:
            # Reset context
            request_context.reset(token)


def log_function_call(logger: logging.Logger | None = None):
    """
    Decorator to log function calls and their results.
    
    Usage:
        @log_function_call()
        def my_function(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(f"api.{func.__module__}")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}")
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator
