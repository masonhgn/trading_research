"""
Universal configuration for the trading research system.
"""

import os
import yaml
import datetime as dt
from typing import Dict, Any, Optional
from pathlib import Path


class TradingConfig:
    """Universal configuration for trading system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, looks for config.yaml in project root.
        """
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'data.bar_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'data.bar_size')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self._config, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Error saving configuration: {e}")
    
    @property
    def DATA_CONFIG(self) -> Dict[str, Any]:
        """Get data layer configuration."""
        return self._config.get('data', {})
    
    @property
    def STRATEGY_CONFIG(self) -> Dict[str, Any]:
        """Get strategy layer configuration."""
        return self._config.get('strategy', {})
    
    @property
    def EXECUTION_CONFIG(self) -> Dict[str, Any]:
        """Get execution layer configuration."""
        return self._config.get('execution', {})
    
    @property
    def ANALYSIS_CONFIG(self) -> Dict[str, Any]:
        """Get analysis layer configuration."""
        return self._config.get('analysis', {})
    
    @property
    def INFRASTRUCTURE_CONFIG(self) -> Dict[str, Any]:
        """Get infrastructure configuration."""
        return self._config.get('infrastructure', {})
    
    @property
    def ENVIRONMENT_CONFIG(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self._config.get('environment', {})
    
    def validate_config(self) -> None:
        """Validate the configuration and print warnings for potential issues."""
        data_config = self.DATA_CONFIG
        
        if 'bar_size' not in data_config:
            print("WARNING: bar_size not found in configuration")
            return
        
        if 'valid_bar_sizes' in data_config and data_config['bar_size'] not in data_config['valid_bar_sizes']:
            print(f"WARNING: Invalid bar_size '{data_config['bar_size']}'. Valid options: {data_config['valid_bar_sizes']}")
        
        if 'signal_frequency' in data_config and data_config['bar_size'] != data_config['signal_frequency']:
            print(f"WARNING: bar_size ({data_config['bar_size']}) differs from signal_frequency ({data_config['signal_frequency']})")
        
        trading_hours = data_config.get('trading_hours', {})
        print(f"✓ Using bar size: {data_config['bar_size']}")
        print(f"✓ Signal frequency: {data_config.get('signal_frequency', 'Not set')}")
        print(f"✓ Trading hours: {trading_hours.get('start', 'Not set')} - {trading_hours.get('end', 'Not set')} {trading_hours.get('timezone', 'Not set')}")
    
    def get_periods_per_year(self) -> int:
        """Calculate periods per year based on bar size configuration."""
        bar_size = self.get('data.bar_size', '1 min')
        
        if 'min' in bar_size:
            if bar_size == '1 min':
                return 252 * 6.5 * 60  # 252 days * 6.5 hours * 60 minutes
            elif bar_size == '5 mins':
                return 252 * 6.5 * 12  # 252 days * 6.5 hours * 12 five-minute periods
            elif bar_size == '15 mins':
                return 252 * 6.5 * 4   # 252 days * 6.5 hours * 4 fifteen-minute periods
            else:
                return 252  # Default to daily
        else:
            return 252  # Default to daily
    
    def get_strategy_config(self, strategy_name: str = 'etf_arbitrage') -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        return self.STRATEGY_CONFIG.get(strategy_name, {})
    
    def update_strategy_config(self, strategy_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a specific strategy."""
        if strategy_name not in self.STRATEGY_CONFIG:
            self.STRATEGY_CONFIG[strategy_name] = {}
        
        self.STRATEGY_CONFIG[strategy_name].update(updates)
    
    def is_paper_trading(self) -> bool:
        """Check if system is configured for paper trading."""
        return self.get('environment.mode', 'paper') == 'paper'
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('environment.debug', False)
    
    def is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.get('environment.verbose', True)


# Global config instance
config = TradingConfig()
