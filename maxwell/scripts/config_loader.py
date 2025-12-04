"""
Configuration loader for maxwell package.

Loads configuration from YAML files with environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config file (default: maxwell/config.yaml)
        """
        if config_path is None:
            # Default to package config.yaml
            config_path = Path(__file__).parent.parent / 'config.yaml'

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path.exists():
            # Return default configuration
            return self._default_config()

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Apply environment variable overrides
        config = self._apply_env_overrides(config)

        return config

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'algorithm': {
                'lambda_stream': 0.5,
                'coherence_threshold': 1.0,
                'max_iterations': 1000,
                'allow_revisitation': True,
                'max_compound_order': 5
            },
            'segmentation': {
                'method': 'slic',
                'n_segments': 50,
                'compactness': 10.0,
                'sigma': 1.0
            },
            'validation': {
                'physical': {
                    'temperature': 310.0
                }
            },
            'paths': {
                'output': 'results',
                'cache': '.cache',
                'data': 'data'
            }
        }

    def _apply_env_overrides(self, config: Dict) -> Dict:
        """Apply environment variable overrides."""
        # Check for MAXWELL_* environment variables
        for key, value in os.environ.items():
            if key.startswith('MAXWELL_'):
                # Convert MAXWELL_ALGORITHM_LAMBDA_STREAM to ['algorithm']['lambda_stream']
                parts = key[8:].lower().split('_')  # Remove MAXWELL_ prefix

                # Navigate config dict
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set value (attempt type conversion)
                try:
                    current[parts[-1]] = float(value)
                except ValueError:
                    try:
                        current[parts[-1]] = int(value)
                    except ValueError:
                        if value.lower() in ['true', 'false']:
                            current[parts[-1]] = value.lower() == 'true'
                        else:
                            current[parts[-1]] = value

        return config

    def get(self, key: str, default=None):
        """
        Get configuration value.

        Args:
            key: Configuration key (dot-separated, e.g., 'algorithm.lambda_stream')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        parts = key.split('.')
        current = self.config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key: Configuration key (dot-separated)
            value: Value to set
        """
        parts = key.split('.')
        current = self.config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def save(self, path: str = None):
        """
        Save configuration to file.

        Args:
            path: Output path (default: original config path)
        """
        if path is None:
            path = self.config_path
        else:
            path = Path(path)

        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to config file (optional)

    Returns:
        Config instance
    """
    global _config

    if _config is None or config_path is not None:
        _config = Config(config_path)

    return _config
