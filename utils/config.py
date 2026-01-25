"""
Configuration management utilities for the benchmark framework.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing configuration parameters.
        
    Raises:
        ValueError: If the file extension is not .json or .yaml/.yml.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file type: {ext}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a JSON or YAML file.
    
    Args:
        config: Dictionary containing configuration parameters.
        config_path: Path where the configuration will be saved.
        
    Raises:
        ValueError: If the file extension is not .json or .yaml/.yml.
    """
    _, ext = os.path.splitext(config_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    if ext.lower() == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    elif ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported configuration file type: {ext}")


class ConfigManager:
    """
    Manager for handling configuration across the benchmark framework.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, an empty
                        configuration is created.
        """
        self.config = {}
        if config_path:
            self.config = load_config(config_path)
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key.
            default: Default value to return if key is not found.
            
        Returns:
            The value for the given key, or the default value if not found.
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key.
            value: Value to set.
        """
        self.config[key] = value
        
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            config_dict: Dictionary of configuration key-value pairs.
        """
        self.config.update(config_dict)
        
    def save(self, config_path: str) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path where the configuration will be saved.
        """
        save_config(self.config, config_path)
