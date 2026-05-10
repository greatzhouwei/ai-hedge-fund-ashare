"""Backtesting package: interfaces and shared types for refactoring.

This module defines the public contracts (Protocols/ABCs) for the
backtesting subsystem. Implementations can live elsewhere and be
introduced gradually without changing existing behavior.
"""

from .types import (
    ActionLiteral,
    AgentDecision,
    AgentDecisions,
    AgentOutput,
    AgentSignals,
    PerformanceMetrics,
    PortfolioSnapshot,
    PortfolioValuePoint,
    PositionState,
    PriceDataFrame,
    TickerRealizedGains,
)

try:
    from .portfolio import Portfolio
    from .trader import TradeExecutor
    from .metrics import PerformanceMetricsCalculator
    from .controller import AgentController
    from .engine import BacktestEngine
    from .valuation import calculate_portfolio_value, compute_exposures
    from .output import OutputBuilder
except ImportError:
    Portfolio = None  # type: ignore[misc,assignment]
    TradeExecutor = None  # type: ignore[misc,assignment]
    PerformanceMetricsCalculator = None  # type: ignore[misc,assignment]
    AgentController = None  # type: ignore[misc,assignment]
    BacktestEngine = None  # type: ignore[misc,assignment]
    calculate_portfolio_value = None  # type: ignore[misc,assignment]
    compute_exposures = None  # type: ignore[misc,assignment]
    OutputBuilder = None  # type: ignore[misc,assignment]

__all__ = [
    # Types
    "ActionLiteral",
    "AgentDecision",
    "AgentDecisions",
    "AgentOutput",
    "AgentSignals",
    "PerformanceMetrics",
    "PortfolioSnapshot",
    "PortfolioValuePoint",
    "PositionState",
    "PriceDataFrame",
    "TickerRealizedGains",
    # Interfaces
    "Portfolio",
    "TradeExecutor",
    "PerformanceMetricsCalculator",
    "AgentController",
    "BacktestEngine",
    "calculate_portfolio_value",
    "compute_exposures",
    "OutputBuilder",
]


