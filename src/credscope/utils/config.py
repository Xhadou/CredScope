"""Configuration utilities for CredScope"""
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config

def setup_paths(config: dict) -> None:
    """Create necessary directories based on configuration
    
    Args:
        config: Configuration dictionary
    """
    paths = [
        config['data']['raw_path'],
        config['data']['processed_path'],
        config['data']['features_path'],
        'models',
        'mlruns'
    ]
    
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
    
    logger.info("All project directories verified/created")