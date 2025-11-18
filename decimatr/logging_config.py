import json
import uuid
import sys
import datetime
from typing import Any, Dict
import numpy as np
from loguru import logger

class NumpySafeJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle common NumPy types and Python objects"""
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, type):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def flatten_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested extras to avoid nesting issues."""
    if not isinstance(extra, dict):
        return extra
        
    flattened = {}
    for key, value in extra.items():
        if key == 'extra' and isinstance(value, dict):
            # If we find a nested 'extra', merge it with the top level
            nested_extra = flatten_extra(value)
            for nested_key, nested_value in nested_extra.items():
                flattened[nested_key] = nested_value
        else:
            flattened[key] = value
    return flattened

def custom_format(record):
    """Process record into a formatted string for both console and file logging."""
    # Get timestamp and level
    timestamp = record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    level = record["level"].name
    message = record["message"]
    
    # Build base log entry
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "logger": record["name"],
        "file": record["file"].name,
        "line": record["line"],
        "function": record["function"]
    }
    
    # Process extra fields if present, flattening nested extras
    raw_extra = record.get("extra", {})
    extra = flatten_extra(raw_extra)
    
    if extra:
        # Directly include first-level extra fields
        for key, value in extra.items():
            if key not in log_entry:  # Avoid overwriting standard fields
                log_entry[key] = value
    
    # Process exception information
    if record["exception"]:
        log_entry["exception"] = str(record["exception"])
    
    # For console output, create a readable format
    component = extra.get("component_name", "")
    operation = extra.get("operation", "")
    outcome = extra.get("outcome", "")
    
    console_msg = f"{timestamp} - {level} - "
    
    # Add component details if present
    if component:
        console_msg += f"{component}"
        if operation:
            console_msg += f".{operation}"
        if outcome:
            console_msg += f" ({outcome})"
        console_msg += " - "
    
    # Add the message
    console_msg += message
    
    # Add metadata if present, truncating if too long
    if "relevant_metadata" in extra:
        metadata = str(extra["relevant_metadata"])
        if len(metadata) > 100:
            metadata = metadata[:97] + "..."
        console_msg += f" - Meta: {metadata}"
    
    # For JSON output
    json_msg = json.dumps(log_entry, cls=NumpySafeJsonEncoder)
    
    return console_msg, json_msg

def setup_logging(logger_name=None, default_level="INFO"):
    """Configure the logger with console and file handlers."""
    # Remove any existing handlers
    logger.remove()
    
    # Custom format function that returns both console and JSON formats
    def format_func(record):
        console_format, json_format = custom_format(record)
        
        # For console output
        print(console_format, file=sys.stderr)
        
        # For file output
        with open("data_pipeline.log", "a") as f:
            f.write(json_format + "\n")
        
        # Return something minimal for loguru's internal handler
        return ""
    
    # Add a simple handler that triggers our custom formatter
    logger.add(lambda message: None, level=default_level, format=format_func)
    
    # Create a named logger if requested
    if logger_name:
        return logger.bind(name=logger_name)
    return logger

if __name__ == '__main__':
    # Example Usage:
    test_logger = setup_logging()

    # Example log with standard fields
    test_logger.info("This is a standard info message.")
    test_logger.warning("This is a standard warning message.")

    # Example log with custom dictionary
    example_metadata = {
        "frame_id": "frame_123",
        "blur_score_calculated": 0.9,
        "threshold_used": 0.7,
        "decision_is_blurred": True
    }
    log_dict = {
        "component_name": "ExampleComponent",
        "operation": "example_operation",
        "outcome": "example_success",
        "event_id": str(uuid.uuid4()),
        "session_id": "test_session_001",
        "relevant_metadata": example_metadata
    }
    test_logger.info("Example component operation finished.", extra=log_dict)

    # Test with nested extra
    nested_log_dict = {
        "extra": {
            "component_name": "NestedComponent",
            "operation": "nested_operation",
            "outcome": "nested_success",
            "relevant_metadata": {"nested": "value"}
        }
    }
    test_logger.info("Testing with nested extra structure", extra=nested_log_dict)

    # Exception test
    try:
        1 / 0
    except ZeroDivisionError:
        test_logger.exception("An error occurred", extra={
            "component_name": "ErrorHandler", 
            "operation": "divide_by_zero", 
            "outcome": "exception_caught", 
            "relevant_metadata": {"details": "Attempted to divide by zero."}
        })