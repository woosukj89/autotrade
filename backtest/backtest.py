import random
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from strategies.strategy import Strategy, Portfolio, Position, ExecutionContext

# Map user-facing period codes to pandas frequency strings.
FREQ_MAP = {
    "m": "min",   # minute
    "h": "h",     # hour
    "d": "B",     # business day
    "M": "MS",    # month (first business day)
    "q": "QS",    # quarter start
    "y": "YS",    # year start
}


# ── result data structures ────────────────────────────────────────────

@dataclass
class TradeRecord:
    """One executed buy or sell."""
    date: datetime
    ticker: str
    action: str          # "buy" or "sell"
    shares: float
    market_price: float
    executed_price: float
    total: float


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a single point in time."""
    date: datetime
    cash: float
    positions_value: float
    total_value: float
    positions: dict[str, float]   # ticker → shares held


class BacktestResult:
    """Collected output of a backtest run."""

    RISK_FREE_RATE = 0.04  # 4% annual risk-free rate assumption

    def __init__(self):
        self.snapshots: list[PortfolioSnapshot] = []
        self.trades: list[TradeRecord] = []
        self.final_portfolio: Optional[Portfolio] = None
        self.benchmark_values: list[float] = []  # benchmark portfolio value over time
        self.benchmark_ticker: str = "SPY"

    # ── scalar metrics ────────────────────────────────────────────────

    @property
    def initial_value(self) -> float:
        return self.snapshots[0].total_value if self.snapshots else 0.0

    @property
    def final_value(self) -> float:
        return self.snapshots[-1].total_value if self.snapshots else 0.0

    @property
    def total_return(self) -> float:
        if self.initial_value == 0:
            return 0.0
        return (self.final_value - self.initial_value) / self.initial_value

    @property
    def cagr(self) -> float:
        if len(self.snapshots) < 2 or self.initial_value <= 0:
            return 0.0
        years = (self.snapshots[-1].date - self.snapshots[0].date).days / 365.25
        if years <= 0:
            return 0.0
        return (self.final_value / self.initial_value) ** (1.0 / years) - 1.0

    @property
    def max_drawdown(self) -> float:
        if not self.snapshots:
            return 0.0
        peak = self.snapshots[0].total_value
        worst = 0.0
        for snap in self.snapshots:
            if snap.total_value > peak:
                peak = snap.total_value
            dd = (peak - snap.total_value) / peak if peak > 0 else 0.0
            if dd > worst:
                worst = dd
        return worst

    # ── export helpers ────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "date": s.date,
                "cash": s.cash,
                "positions_value": s.positions_value,
                "total_value": s.total_value,
            }
            for s in self.snapshots
        ])

    def trades_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "date": t.date,
                "ticker": t.ticker,
                "action": t.action,
                "shares": t.shares,
                "market_price": t.market_price,
                "executed_price": t.executed_price,
                "total": t.total,
            }
            for t in self.trades
        ])

    def positions_over_time(self) -> pd.DataFrame:
        """Return a DataFrame showing portfolio positions at each snapshot.

        Columns: date, ticker, shares, plus cash and total_value.
        """
        records = []
        for snap in self.snapshots:
            # Add a row for cash.
            records.append({
                "date": snap.date,
                "ticker": "_CASH",
                "shares": snap.cash,
                "value": snap.cash,
            })
            # Add rows for each position.
            for ticker, shares in snap.positions.items():
                records.append({
                    "date": snap.date,
                    "ticker": ticker,
                    "shares": shares,
                    "value": None,  # Would need price to compute
                })
        return pd.DataFrame(records)

    def portfolio_summary_over_time(self) -> pd.DataFrame:
        """Return a DataFrame with portfolio value and positions count over time."""
        records = []
        for i, snap in enumerate(self.snapshots):
            bench_val = self.benchmark_values[i] if i < len(self.benchmark_values) else None
            records.append({
                "date": snap.date,
                "cash": snap.cash,
                "positions_value": snap.positions_value,
                "total_value": snap.total_value,
                "num_positions": len(snap.positions),
                "positions": ", ".join(snap.positions.keys()) if snap.positions else "None",
                "benchmark_value": bench_val,
            })
        return pd.DataFrame(records)

    def summary(self) -> dict:
        return {
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "total_trades": len(self.trades),
        }

    # ── benchmark analytics ────────────────────────────────────────────

    def _daily_returns(self, values: list[float]) -> np.ndarray:
        """Compute daily returns from a list of portfolio values."""
        arr = np.array(values)
        return np.diff(arr) / arr[:-1]

    @property
    def strategy_returns(self) -> np.ndarray:
        """Daily returns of the strategy."""
        values = [s.total_value for s in self.snapshots]
        return self._daily_returns(values)

    @property
    def benchmark_returns(self) -> np.ndarray:
        """Daily returns of the benchmark."""
        return self._daily_returns(self.benchmark_values)

    @property
    def benchmark_total_return(self) -> float:
        """Total return of the benchmark."""
        if not self.benchmark_values or self.benchmark_values[0] == 0:
            return 0.0
        return (self.benchmark_values[-1] - self.benchmark_values[0]) / self.benchmark_values[0]

    @property
    def benchmark_cagr(self) -> float:
        """CAGR of the benchmark."""
        if len(self.snapshots) < 2 or not self.benchmark_values:
            return 0.0
        years = (self.snapshots[-1].date - self.snapshots[0].date).days / 365.25
        if years <= 0 or self.benchmark_values[0] <= 0:
            return 0.0
        return (self.benchmark_values[-1] / self.benchmark_values[0]) ** (1.0 / years) - 1.0

    @property
    def beta(self) -> float:
        """Beta of the strategy relative to the benchmark."""
        strat_ret = self.strategy_returns
        bench_ret = self.benchmark_returns
        if len(strat_ret) < 2 or len(bench_ret) < 2:
            return 0.0
        # Align lengths
        min_len = min(len(strat_ret), len(bench_ret))
        strat_ret = strat_ret[:min_len]
        bench_ret = bench_ret[:min_len]
        cov = np.cov(strat_ret, bench_ret)[0, 1]
        var = np.var(bench_ret)
        return cov / var if var > 0 else 0.0

    @property
    def alpha(self) -> float:
        """Annualized Jensen's alpha."""
        if len(self.snapshots) < 2:
            return 0.0
        years = (self.snapshots[-1].date - self.snapshots[0].date).days / 365.25
        if years <= 0:
            return 0.0
        rf = self.RISK_FREE_RATE
        strat_annual = self.cagr
        bench_annual = self.benchmark_cagr
        return strat_annual - (rf + self.beta * (bench_annual - rf))

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio of the strategy."""
        strat_ret = self.strategy_returns
        if len(strat_ret) < 2:
            return 0.0
        daily_rf = self.RISK_FREE_RATE / 252
        excess = strat_ret - daily_rf
        if np.std(excess) == 0:
            return 0.0
        return np.mean(excess) / np.std(excess) * np.sqrt(252)

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino ratio (downside deviation)."""
        strat_ret = self.strategy_returns
        if len(strat_ret) < 2:
            return 0.0
        daily_rf = self.RISK_FREE_RATE / 252
        excess = strat_ret - daily_rf
        downside = excess[excess < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0
        return np.mean(excess) / np.std(downside) * np.sqrt(252)

    @property
    def information_ratio(self) -> float:
        """Information ratio (excess return / tracking error)."""
        strat_ret = self.strategy_returns
        bench_ret = self.benchmark_returns
        if len(strat_ret) < 2 or len(bench_ret) < 2:
            return 0.0
        min_len = min(len(strat_ret), len(bench_ret))
        excess = strat_ret[:min_len] - bench_ret[:min_len]
        if np.std(excess) == 0:
            return 0.0
        return np.mean(excess) / np.std(excess) * np.sqrt(252)

    @property
    def volatility(self) -> float:
        """Annualized volatility of strategy returns."""
        strat_ret = self.strategy_returns
        if len(strat_ret) < 2:
            return 0.0
        return np.std(strat_ret) * np.sqrt(252)

    @property
    def benchmark_volatility(self) -> float:
        """Annualized volatility of benchmark returns."""
        bench_ret = self.benchmark_returns
        if len(bench_ret) < 2:
            return 0.0
        return np.std(bench_ret) * np.sqrt(252)

    def full_stats(self) -> dict:
        """Return comprehensive statistics including benchmark comparison."""
        return {
            # Strategy metrics
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "total_trades": len(self.trades),
            # Benchmark comparison
            "benchmark": self.benchmark_ticker,
            "benchmark_total_return": self.benchmark_total_return,
            "benchmark_cagr": self.benchmark_cagr,
            "benchmark_volatility": self.benchmark_volatility,
            # Relative metrics
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
            "excess_return": self.total_return - self.benchmark_total_return,
        }

    def plot(self, title: str = "Backtest Results", save_path: str = None,
             interactive: bool = True) -> None:
        """Plot portfolio value over time with benchmark comparison and stats.

        Args:
            title: Chart title
            save_path: Path to save the chart (if None, displays interactively)
            interactive: If True, enables hover to show portfolio positions
        """
        if not self.snapshots:
            print("No data to plot.")
            return

        dates = [s.date for s in self.snapshots]
        values = [s.total_value for s in self.snapshots]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # ── Top chart: equity curves ──────────────────────────────────
        line1, = ax1.plot(dates, values, label="Strategy", linewidth=2, color="#2E86AB")
        # Add scatter points for hover detection
        scatter = ax1.scatter(dates, values, s=20, color="#2E86AB", alpha=0.6, zorder=5)

        if self.benchmark_values:
            ax1.plot(
                dates[: len(self.benchmark_values)],
                self.benchmark_values,
                label=f"Benchmark ({self.benchmark_ticker})",
                linewidth=2,
                color="#A23B72",
                linestyle="--",
            )
        ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(dates[0], dates[-1])

        # ── Bottom chart: drawdown ────────────────────────────────────
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak * 100
        ax2.fill_between(dates, drawdown, 0, color="#E74C3C", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(True, alpha=0.3)

        # ── Stats box ─────────────────────────────────────────────────
        stats = self.full_stats()
        stats_text = (
            f"Strategy Performance\n"
            f"{'─' * 28}\n"
            f"Total Return:    {stats['total_return']*100:>8.2f}%\n"
            f"CAGR:            {stats['cagr']*100:>8.2f}%\n"
            f"Volatility:      {stats['volatility']*100:>8.2f}%\n"
            f"Max Drawdown:    {stats['max_drawdown']*100:>8.2f}%\n"
            f"Sharpe Ratio:    {stats['sharpe_ratio']:>8.2f}\n"
            f"Sortino Ratio:   {stats['sortino_ratio']:>8.2f}\n"
            f"Total Trades:    {stats['total_trades']:>8}\n"
            f"\n"
            f"Benchmark ({stats['benchmark']})\n"
            f"{'─' * 28}\n"
            f"Total Return:    {stats['benchmark_total_return']*100:>8.2f}%\n"
            f"CAGR:            {stats['benchmark_cagr']*100:>8.2f}%\n"
            f"Volatility:      {stats['benchmark_volatility']*100:>8.2f}%\n"
            f"\n"
            f"Relative Metrics\n"
            f"{'─' * 28}\n"
            f"Alpha:           {stats['alpha']*100:>8.2f}%\n"
            f"Beta:            {stats['beta']:>8.2f}\n"
            f"Info Ratio:      {stats['information_ratio']:>8.2f}\n"
            f"Excess Return:   {stats['excess_return']*100:>8.2f}%"
        )

        props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray")
        ax1.text(
            1.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace", bbox=props
        )

        # ── Interactive hover for portfolio positions ─────────────────
        if interactive and not save_path:
            # Create annotation for hover
            annot = ax1.annotate(
                "", xy=(0, 0), xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", alpha=0.95),
                fontsize=8, fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
            )
            annot.set_visible(False)

            def get_position_text(idx):
                """Generate position text for a given snapshot index."""
                snap = self.snapshots[idx]
                date_str = snap.date.strftime("%Y-%m-%d")

                # Format positions
                positions = snap.positions
                num_pos = len(positions)

                text = f"Date: {date_str}\n"
                text += f"{'─' * 30}\n"
                text += f"Total Value: ${snap.total_value:,.0f}\n"
                text += f"Cash: ${snap.cash:,.0f}\n"
                text += f"Positions: ${snap.positions_value:,.0f}\n"
                text += f"{'─' * 30}\n"
                text += f"Holdings ({num_pos}):\n"

                # Show positions (limit to 12 for readability)
                pos_list = list(positions.items())
                for ticker, shares in pos_list[:12]:
                    text += f"  {ticker:<6} {shares:>8.0f} shares\n"
                if len(pos_list) > 12:
                    text += f"  ... and {len(pos_list) - 12} more"

                return text

            def on_hover(event):
                """Handle mouse hover events."""
                if event.inaxes != ax1:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                    return

                cont, ind = scatter.contains(event)
                if cont:
                    idx = ind["ind"][0]
                    pos = scatter.get_offsets()[idx]
                    annot.xy = pos
                    annot.set_text(get_position_text(idx))
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", on_hover)
            print("Hover over data points to see portfolio positions.")

        plt.tight_layout()
        plt.subplots_adjust(right=0.78)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Chart saved to {save_path}")
        else:
            plt.show()

    def plot_interactive(self, title: str = "Backtest Results", save_path: str = None) -> None:
        """Plot with Plotly for better interactivity (if available).

        Falls back to matplotlib if Plotly is not installed.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Using matplotlib instead.")
            print("Install with: pip install plotly")
            return self.plot(title=title, save_path=save_path)

        if not self.snapshots:
            print("No data to plot.")
            return

        dates = [s.date for s in self.snapshots]
        values = [s.total_value for s in self.snapshots]

        # Build hover text for each point
        hover_texts = []
        for snap in self.snapshots:
            positions = snap.positions
            pos_text = "<br>".join([f"  {t}: {s:.0f}" for t, s in list(positions.items())[:15]])
            if len(positions) > 15:
                pos_text += f"<br>  ... +{len(positions) - 15} more"

            text = (
                f"<b>Date:</b> {snap.date.strftime('%Y-%m-%d')}<br>"
                f"<b>Total Value:</b> ${snap.total_value:,.0f}<br>"
                f"<b>Cash:</b> ${snap.cash:,.0f}<br>"
                f"<b>Positions Value:</b> ${snap.positions_value:,.0f}<br>"
                f"<b>Holdings ({len(positions)}):</b><br>{pos_text}"
            )
            hover_texts.append(text)

        # Calculate drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak * 100

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.75, 0.25],
            subplot_titles=("Portfolio Value", "Drawdown")
        )

        # Strategy line
        fig.add_trace(
            go.Scatter(
                x=dates, y=values,
                mode='lines+markers',
                name='Strategy',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=6),
                hovertext=hover_texts,
                hoverinfo='text',
            ),
            row=1, col=1
        )

        # Benchmark line
        if self.benchmark_values:
            fig.add_trace(
                go.Scatter(
                    x=dates[:len(self.benchmark_values)],
                    y=self.benchmark_values,
                    mode='lines',
                    name=f'Benchmark ({self.benchmark_ticker})',
                    line=dict(color='#A23B72', width=2, dash='dash'),
                    hovertemplate='%{x}<br>Value: $%{y:,.0f}<extra></extra>',
                ),
                row=1, col=1
            )

        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=dates, y=drawdown,
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='#E74C3C', width=1),
                fillcolor='rgba(231, 76, 60, 0.3)',
                hovertemplate='%{x}<br>Drawdown: %{y:.1f}%<extra></extra>',
            ),
            row=2, col=1
        )

        # Add stats annotation
        stats = self.full_stats()
        stats_text = (
            f"<b>Strategy Performance</b><br>"
            f"Total Return: {stats['total_return']*100:.2f}%<br>"
            f"CAGR: {stats['cagr']*100:.2f}%<br>"
            f"Max Drawdown: {stats['max_drawdown']*100:.2f}%<br>"
            f"Sharpe: {stats['sharpe_ratio']:.2f}<br>"
            f"<br><b>vs Benchmark</b><br>"
            f"Alpha: {stats['alpha']*100:.2f}%<br>"
            f"Beta: {stats['beta']:.2f}"
        )

        fig.add_annotation(
            x=1.02, y=0.98,
            xref="paper", yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            borderpad=6,
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=700,
            width=1200,
            hovermode='closest',
            showlegend=True,
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
            margin=dict(r=200),
        )

        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            print(f"Chart saved to {save_path}")
        else:
            fig.show()


# ── main engine ───────────────────────────────────────────────────────

class Backtest:
    """Backtesting engine.

    Repeatedly calls ``strategy.execute()`` at each time step, diffs the
    returned portfolio against the live portfolio, executes the implied
    trades (with optional buffer-pricing slippage), and records a full
    history of snapshots and trades.

    Price data is fetched lazily from *yfinance* the first time a ticker
    is referenced, then cached for the rest of the run.  Fundamental data
    is read from the local SQLite database.

    Note: intra-day frequencies ("m", "h") require yfinance intra-day
    data which is only available for recent history.  For multi-year
    backtests use "d" or longer.
    """

    def __init__(self, db_path: str = "fundamentals.sqlite"):
        self.db_path = db_path
        self._price_cache: dict[str, pd.DataFrame] = {}
        self._start_date: Optional[datetime] = None
        self._end_date: Optional[datetime] = None
        self._current_date: Optional[datetime] = None

    # ── price helpers ─────────────────────────────────────────────────

    def _ensure_prices(self, ticker: str) -> None:
        if ticker in self._price_cache:
            return
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(
                    ticker,
                    start=self._start_date.strftime("%Y-%m-%d"),
                    end=self._end_date.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                    timeout=10,
                )
            if df is None or df.empty:
                self._price_cache[ticker] = pd.DataFrame()
                return
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            self._price_cache[ticker] = df
        except Exception:
            self._price_cache[ticker] = pd.DataFrame()

    def _get_price(self, ticker: str, date: datetime) -> Optional[float]:
        self._ensure_prices(ticker)
        df = self._price_cache.get(ticker)
        if df is None or df.empty:
            return None
        valid = df.loc[:pd.Timestamp(date)]
        if valid.empty:
            return None
        return float(valid["Close"].iloc[-1])

    def _get_historical(self, ticker: str, date: datetime,
                        periods: int) -> Optional[pd.DataFrame]:
        self._ensure_prices(ticker)
        df = self._price_cache.get(ticker)
        if df is None or df.empty:
            return None
        valid = df.loc[:pd.Timestamp(date)]
        return valid.tail(periods).copy()

    # ── fundamentals helper ───────────────────────────────────────────

    def _get_fundamentals(self, ticker: str, date: datetime,
                          field_name: str = None) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            date_str = date.strftime("%Y-%m-%d")
            if field_name:
                query = (
                    "SELECT fy, date, field, value "
                    "FROM fundamentals "
                    "WHERE ticker = ? AND field = ? AND date <= ? "
                    "ORDER BY date DESC"
                )
                df = pd.read_sql_query(query, conn, params=(ticker, field_name, date_str))
            else:
                query = (
                    "SELECT fy, date, statement_type, field, value "
                    "FROM fundamentals "
                    "WHERE ticker = ? AND date <= ? "
                    "ORDER BY date DESC"
                )
                df = pd.read_sql_query(query, conn, params=(ticker, date_str))
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()

    # ── buffer pricing ────────────────────────────────────────────────

    @staticmethod
    def _apply_buffer(price: float, side: str, buffer_pricing: int) -> float:
        if buffer_pricing <= 0:
            return price
        offset = random.uniform(0, buffer_pricing)
        if side == "buy":
            return price + offset
        return max(0.01, price - offset)

    # ── portfolio valuation ───────────────────────────────────────────

    def _positions_value(self, portfolio: Portfolio) -> float:
        total = 0.0
        for ticker, pos in portfolio.positions.items():
            price = self._get_price(ticker, self._current_date)
            total += pos.shares * (price if price is not None else pos.avg_cost)
        return total

    # ── trade execution ───────────────────────────────────────────────

    def _execute_trades(self, current: Portfolio, desired: Portfolio,
                        buffer_pricing: int) -> list[TradeRecord]:
        """Diff *current* vs *desired* and mutate *current* in place.

        Sells are processed first to free up cash for buys.  If after
        buffer-pricing a buy would exceed available cash the share count
        is reduced accordingly.
        """
        trades: list[TradeRecord] = []

        # ---- sells ----
        for ticker in list(current.positions):
            cur_shares = current.positions[ticker].shares
            des_shares = desired.positions[ticker].shares if ticker in desired.positions else 0.0

            if des_shares >= cur_shares:
                continue

            sell_shares = cur_shares - des_shares
            market_price = self._get_price(ticker, self._current_date)
            if market_price is None:
                continue

            exec_price = self._apply_buffer(market_price, "sell", buffer_pricing)
            proceeds = sell_shares * exec_price
            current.cash += proceeds

            if des_shares <= 0:
                del current.positions[ticker]
            else:
                current.positions[ticker].shares = des_shares

            trades.append(TradeRecord(
                date=self._current_date,
                ticker=ticker,
                action="sell",
                shares=sell_shares,
                market_price=market_price,
                executed_price=exec_price,
                total=proceeds,
            ))

        # ---- buys ----
        for ticker, des_pos in desired.positions.items():
            cur_shares = current.positions[ticker].shares if ticker in current.positions else 0.0

            if des_pos.shares <= cur_shares:
                continue

            buy_shares = des_pos.shares - cur_shares
            market_price = self._get_price(ticker, self._current_date)
            if market_price is None:
                continue

            exec_price = self._apply_buffer(market_price, "buy", buffer_pricing)
            cost = buy_shares * exec_price

            # Reduce order if cash is insufficient.
            if cost > current.cash:
                buy_shares = current.cash / exec_price
                cost = buy_shares * exec_price
                if buy_shares <= 0:
                    continue

            current.cash -= cost

            if ticker in current.positions:
                old = current.positions[ticker]
                new_total = old.shares + buy_shares
                old_cost_basis = old.shares * old.avg_cost
                new_cost_basis = buy_shares * exec_price
                current.positions[ticker] = Position(
                    ticker=ticker,
                    shares=new_total,
                    avg_cost=(old_cost_basis + new_cost_basis) / new_total,
                )
            else:
                current.positions[ticker] = Position(
                    ticker=ticker,
                    shares=buy_shares,
                    avg_cost=exec_price,
                )

            trades.append(TradeRecord(
                date=self._current_date,
                ticker=ticker,
                action="buy",
                shares=buy_shares,
                market_price=market_price,
                executed_price=exec_price,
                total=cost,
            ))

        return trades

    # ── public API ────────────────────────────────────────────────────

    def backtest(
        self,
        lookback_years: int = 10,
        end_year: int = None,
        starting_fund: float = 10000,
        buffer_pricing: int = 0,
        strategy: Strategy = None,
        time_period: str = "d",
        benchmark: str = "SPY",
    ) -> BacktestResult:
        """Run a backtest.

        Args:
            lookback_years: Total number of years to simulate.
            end_year: Calendar year to look back from (inclusive).
                      Defaults to the current year.
            starting_fund: Initial cash in USD.
            buffer_pricing: Maximum random slippage in USD applied to each
                            trade.  ``0`` disables slippage.  When > 0 buy
                            prices are raised and sell prices lowered by a
                            uniform-random amount in ``[0, buffer_pricing]``.
            strategy: A :class:`Strategy` instance whose ``execute`` method
                      will be called at every time step.
            time_period: Execution frequency — one of ``"m"`` (minute),
                         ``"h"`` (hour), ``"d"`` (day), ``"M"`` (month),
                         ``"q"`` (quarter), ``"y"`` (year).
            benchmark: Ticker symbol for benchmark comparison (default: "SPY").

        Returns:
            A :class:`BacktestResult` containing portfolio snapshots,
            trade records, benchmark comparison, and summary metrics.
        """
        if strategy is None:
            raise ValueError("A Strategy instance is required.")
        if time_period not in FREQ_MAP:
            raise ValueError(
                f"Invalid time_period '{time_period}'. "
                f"Must be one of: {list(FREQ_MAP.keys())}"
            )

        if end_year is None:
            end_year = datetime.now().year

        self._start_date = datetime(end_year - lookback_years, 1, 1)
        self._end_date = datetime(end_year, 12, 31)
        self._price_cache = {}

        freq = FREQ_MAP[time_period]
        dates = pd.date_range(start=self._start_date, end=self._end_date, freq=freq)

        portfolio = Portfolio(cash=starting_fund)
        result = BacktestResult()
        result.benchmark_ticker = benchmark

        # Pre-fetch benchmark prices and find first available price.
        self._ensure_prices(benchmark)
        benchmark_df = self._price_cache.get(benchmark)
        benchmark_shares = 0.0
        if benchmark_df is not None and not benchmark_df.empty:
            # Use first available price in the period (first trading day).
            first_bench_price = float(benchmark_df["Close"].iloc[0])
            if first_bench_price > 0:
                benchmark_shares = starting_fund / first_bench_price

        for ts in dates:
            self._current_date = ts.to_pydatetime()

            context = ExecutionContext(
                date=self._current_date,
                portfolio=deepcopy(portfolio),
                get_price_fn=self._get_price,
                get_historical_fn=self._get_historical,
                get_fundamentals_fn=self._get_fundamentals,
            )

            desired = strategy.execute(context)

            trades = self._execute_trades(portfolio, desired, buffer_pricing)
            result.trades.extend(trades)

            pos_value = self._positions_value(portfolio)
            result.snapshots.append(PortfolioSnapshot(
                date=self._current_date,
                cash=portfolio.cash,
                positions_value=pos_value,
                total_value=portfolio.cash + pos_value,
                positions={t: p.shares for t, p in portfolio.positions.items()},
            ))

            # Track benchmark value.
            bench_price = self._get_price(benchmark, self._current_date)
            if bench_price is not None and benchmark_shares > 0:
                result.benchmark_values.append(bench_price * benchmark_shares)
            else:
                # Use last known value or starting fund.
                result.benchmark_values.append(
                    result.benchmark_values[-1] if result.benchmark_values else starting_fund
                )

        result.final_portfolio = deepcopy(portfolio)
        return result
