"""
Data Package
=============

Provides data sources and utilities for the autotrade system.

Available Providers:
    - YahooDataProvider: Unified data source using Yahoo Finance API
"""

from .yahoo_data import (
    YahooDataProvider,
    StockFundamentals,
    get_yahoo_provider,
)

__all__ = [
    'YahooDataProvider',
    'StockFundamentals',
    'get_yahoo_provider',
]
