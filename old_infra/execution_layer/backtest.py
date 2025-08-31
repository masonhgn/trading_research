"""
Backtesting execution layer for the trading system.
"""

import datetime as dt
from typing import Dict, Any, Optional
import pandas as pd

from strategy_layer.base import BaseStrategy
from analysis_layer.analyzer import BacktestAnalyzer
from infrastructure.config import config


class BacktestEngine:
    """
    Backtesting engine that orchestrates strategy backtesting with analysis.
    """
    
    def __init__(self, strategy: BaseStrategy):
        """
        Initialize the backtest engine.
        
        Args:
            strategy: Strategy instance to backtest (should be created with data_mode='backtest')
        """
        self.strategy = strategy
        
        # Ensure strategy is in backtest mode
        if self.strategy.data_mode != 'backtest':
            raise ValueError("BacktestEngine requires strategy with data_mode='backtest'")
        
        self.analyzer = BacktestAnalyzer()
    
    async def run_backtest(
        self, 
        duration: str = None, 
        end_datetime: Optional[dt.datetime] = None,
        show_plots: bool = True,
        show_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete backtest with analysis.
        
        Args:
            duration: Duration string for backtest
            end_datetime: End datetime for backtest
            show_plots: Whether to show strategy plots
            show_analysis: Whether to show performance analysis
            
        Returns:
            Dictionary with backtest results and analysis
        """
        # Use config defaults if not specified
        if duration is None:
            duration = config.EXECUTION_CONFIG['backtest']['default_duration']
        if end_datetime is None:
            end_datetime = config.EXECUTION_CONFIG['backtest']['default_end_datetime']
        
        print("="*60)
        print("BACKTEST EXECUTION")
        print("="*60)
        
        # Display configuration summary
        config_summary = self.strategy.get_config_summary()
        print(f"Strategy: {config_summary['strategy_name']}")
        print(f"Bar Size: {config_summary['bar_size']}")
        print(f"Signal Frequency: {config_summary['signal_frequency']}")
        print(f"Trading Hours: {config_summary['trading_hours']['start']} - {config_summary['trading_hours']['end']} {config_summary['trading_hours']['timezone']}")
        print(f"Rolling Window: {config_summary['window']} periods")
        print(f"Dynamic Thresholds: {config_summary['use_dynamic_thresholds']}")
        if config_summary['use_dynamic_thresholds']:
            print(f"Distribution Type: {config_summary['distribution_type']}")
            print(f"Confidence Level: {config_summary['confidence_level']}")
        else:
            print(f"Entry Threshold: {config_summary['entry_threshold']}")
            print(f"Exit Threshold: {config_summary['exit_threshold']}")
        print("="*60)
        
        # Run strategy backtest
        print(f"\nRunning backtest for duration: {duration}")
        df = await self.strategy.run_backtest(duration, end_datetime)
        
        # Run cointegration tests if available
        cointegration_results = None
        if hasattr(self.strategy, 'run_cointegration_tests'):
            cointegration_results = self.strategy.run_cointegration_tests(df)
        
        # Show strategy-specific plots
        if show_plots:
            if hasattr(self.strategy, 'plot_spread'):
                self.strategy.plot_spread(df)
            if hasattr(self.strategy, 'plot_pnl'):
                self.strategy.plot_pnl(df)
            if hasattr(self.strategy, 'plot_distribution_analysis') and config_summary['use_dynamic_thresholds']:
                self.strategy.plot_distribution_analysis(df)
            if hasattr(self.strategy, 'plot_dynamic_thresholds') and config_summary['use_dynamic_thresholds']:
                self.strategy.plot_dynamic_thresholds(df)
        
        # Run performance analysis
        if show_analysis:
            df_analyzed, analysis_summary = self.analyzer.analyze_backtest(
                df, 
                initial_capital=config_summary['initial_capital']
            )
            
            # Print comprehensive performance summary
            self.analyzer.print_summary(analysis_summary)
            
            # Debug return calculations
            self.analyzer.debug_returns(df_analyzed)
            
            # Plot comprehensive performance analysis
            self.analyzer.plot_performance_analysis(df_analyzed, analysis_summary)
        else:
            analysis_summary = None
            df_analyzed = df
        
        # Print cointegration results
        if cointegration_results:
            print("\nCointegration Test Results:")
            print(f"  Statistic = {cointegration_results['coint_score']:.4f}")
            print(f"  p-value   = {cointegration_results['coint_pval']:.4f}")
        
        # Print dynamic threshold statistics if applicable
        if config_summary['use_dynamic_thresholds'] and 'dynamic_entry_threshold' in df.columns:
            entry_thresholds = df['dynamic_entry_threshold'].dropna()
            exit_thresholds = df['dynamic_exit_threshold'].dropna()
            
            print(f"\nDynamic Threshold Statistics:")
            print(f"  Entry threshold - Mean: {entry_thresholds.mean():.3f}, Std: {entry_thresholds.std():.3f}")
            print(f"  Exit threshold - Mean: {exit_thresholds.mean():.3f}, Std: {exit_thresholds.std():.3f}")
            print(f"  Entry threshold range: [{entry_thresholds.min():.3f}, {entry_thresholds.max():.3f}]")
            print(f"  Exit threshold range: [{exit_thresholds.min():.3f}, {exit_thresholds.max():.3f}]")
        elif not config_summary['use_dynamic_thresholds']:
            print(f"\nUsing Fixed Thresholds:")
            print(f"  Entry threshold: {config_summary['entry_threshold']}")
            print(f"  Exit threshold: {config_summary['exit_threshold']}")
        
        return {
            'data': df,
            'analyzed_data': df_analyzed,
            'analysis_summary': analysis_summary,
            'cointegration_results': cointegration_results,
            'config_summary': config_summary
        }
