"""
Configuration loader for the trading system.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    Configuration loader that supports YAML files, environment variables, and defaults.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the config file. If None, will look for config.yaml in project root.
        """
        if config_path is None:
            # Look for config.yaml in the project root
            project_root = self._find_project_root()
            config_path = os.path.join(project_root, "config.yaml")
        
        self.config_path = config_path
        self._config = None
    
    def _find_project_root(self) -> str:
        """
        Find the project root directory by looking for config.yaml or README.md.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Walk up the directory tree looking for project root indicators
        while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
            if (os.path.exists(os.path.join(current_dir, "config.yaml")) or
                os.path.exists(os.path.join(current_dir, "README.md"))):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        
        # Fallback to current working directory
        return os.getcwd()
    
    def load_config(self, reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from file and environment variables.
        
        Args:
            reload: Force reload of configuration even if already loaded
            
        Returns:
            Dictionary containing the configuration
        """
        if self._config is not None and not reload:
            return self._config
        
        # Load default configuration
        default_config = self._get_default_config()
        
        # Load from file if it exists
        file_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                print(f"✓ Loaded configuration from {self.config_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load config from {self.config_path}: {e}")
                print("   Using default configuration")
        else:
            print(f"⚠️  Warning: Config file not found at {self.config_path}")
            print("   Using default configuration")
        
        # Load from environment variables
        env_config = self._load_from_environment()
        
        # Merge configurations (env overrides file, file overrides defaults)
        self._config = self._merge_configs(default_config, file_config, env_config)
        
        return self._config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        """
        return {
            "data": {
                "bar_size": "1 min",
                "signal_frequency": "1 min",
                "trading_hours": {
                    "start": "09:30",
                    "end": "16:00",
                    "timezone": "US/Eastern"
                },
                "valid_bar_sizes": [
                    "1 secs", "5 secs", "10 secs", "30 secs",
                    "1 min", "5 mins", "15 mins", "30 mins",
                    "1 hour", "1 day"
                ]
            },
            "strategy": {
                "etf_arbitrage": {
                    "symbols": ["SPY", "VOO"],
                    "window": 90,
                    "entry_threshold": 1.5,
                    "exit_threshold": 0.2,
                    "use_dynamic_thresholds": False,
                    "confidence_level": 0.95,
                    "distribution_type": "t",
                    "slippage": 0.01,
                    "fee": 0.005,
                    "shares_per_leg": 200,
                    "initial_capital": 10000,
                    "filter_trading_hours": True,
                    "auto_detect_hours": False,
                    "volatility_threshold": 0.001
                }
            },
            "execution": {
                "ib_connection": {
                    "host": "127.0.0.1",
                    "live_port": 4001,
                    "paper_port": 4002,
                    "client_id": 11,
                    "timeout": 20
                },
                "risk_limits": {
                    "max_position_size_pct": 0.1,
                    "max_daily_loss_pct": 0.02,
                    "max_drawdown_pct": 0.05,
                    "max_trades_per_day": 50,
                    "emergency_stop_loss_pct": 0.03
                },
                "backtest": {
                    "default_duration": "1 D",
                    "default_end_datetime": None
                }
            },
            "analysis": {
                "rolling_window": 60,
                "annualization_factor": 252,
                "var_confidence": 0.95,
                "plot_figsize": [15, 6]
            },
            "infrastructure": {
                "notifications": {
                    "telegram_enabled": True,
                    "trade_updates": True,
                    "pnl_updates": True,
                    "system_alerts": True
                },
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "environment": {
                "mode": "paper",
                "debug": False,
                "verbose": True
            }
        }
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        Environment variables should be prefixed with TRADING_ and use underscores.
        Example: TRADING_DATA_BAR_SIZE=5 mins
        """
        env_config = {}
        
        # Helper function to set nested dictionary value
        def set_nested_value(d, keys, value):
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value
        
        for key, value in os.environ.items():
            if key.startswith("TRADING_"):
                # Convert TRADING_DATA_BAR_SIZE to data.bar_size
                config_key = key[8:].lower()  # Remove TRADING_ prefix
                keys = config_key.split("_")
                
                # Convert value to appropriate type
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.lower() == "null":
                    value = None
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "").isdigit():
                    value = float(value)
                elif value.startswith("[") and value.endswith("]"):
                    # Handle simple list format like [1, 2, 3]
                    try:
                        value = eval(value)  # Safe for simple lists
                    except:
                        pass  # Keep as string if eval fails
                
                set_nested_value(env_config, keys, value)
        
        return env_config
    
    def _merge_configs(self, *configs) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        Later configs override earlier ones.
        """
        result = {}
        
        for config in configs:
            if not config:
                continue
            
            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value (e.g., "data.bar_size")
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        config = self.load_config()
        keys = key_path.split(".")
        
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def validate_config(self) -> None:
        """
        Validate the loaded configuration and print warnings for potential issues.
        """
        config = self.load_config()
        
        # Validate data configuration
        data_config = config.get("data", {})
        bar_size = data_config.get("bar_size", "")
        signal_frequency = data_config.get("signal_frequency", "")
        valid_bar_sizes = data_config.get("valid_bar_sizes", [])
        
        if bar_size not in valid_bar_sizes:
            print(f"⚠️  WARNING: Invalid bar_size '{bar_size}'. Valid options: {valid_bar_sizes}")
        
        if bar_size != signal_frequency:
            print(f"⚠️  WARNING: bar_size ({bar_size}) differs from signal_frequency ({signal_frequency})")
        
        # Validate trading hours
        trading_hours = data_config.get("trading_hours", {})
        start_time = trading_hours.get("start", "")
        end_time = trading_hours.get("end", "")
        timezone = trading_hours.get("timezone", "")
        
        print(f"✓ Using bar size: {bar_size}")
        print(f"✓ Signal frequency: {signal_frequency}")
        print(f"✓ Trading hours: {start_time} - {end_time} {timezone}")
        
        # Validate environment mode
        env_config = config.get("environment", {})
        mode = env_config.get("mode", "paper")
        print(f"✓ Environment mode: {mode}")
    
    def save_config(self, config: Dict[str, Any], path: Optional[str] = None) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary to save
            path: Path to save the config file. If None, uses the original config path.
        """
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            print(f"✓ Configuration saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")


# Global config loader instance
config_loader = ConfigLoader()


def load_config(reload: bool = False) -> Dict[str, Any]:
    """
    Load configuration using the global config loader.
    
    Args:
        reload: Force reload of configuration
        
    Returns:
        Configuration dictionary
    """
    return config_loader.load_config(reload)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a specific configuration value using dot notation.
    
    Args:
        key_path: Dot-separated path to the config value
        default: Default value if key doesn't exist
        
    Returns:
        Configuration value
    """
    return config_loader.get(key_path, default)
