"""
Rolling-window backtester for MIRAGE++ portfolio and alpha models.

Provides two classes:

    RollingPortfolioBacktest
        Re-fits a PortfolioAllocator every `refit_freq` periods using a
        sliding window of `window` observations.  Tracks weights, turnover,
        realised returns, and diversification metrics over time.

        Supports explicit transaction cost modelling via the `cost_bps`
        parameter (cost in basis points per unit of L1 turnover).  The
        cost-adjusted return for each period is:

            r_net = r_gross - cost_bps * 1e-4 * L1_turnover

        where L1_turnover = sum_i |w_i_new - w_i_old| is the fraction of
        portfolio value that changes hands.  Setting cost_bps=0 (default)
        reproduces the gross-return backtest.

    RollingSignalBacktest
        Same rolling-window logic for an AlphaSignalCombiner: predicts
        forward returns from signal matrix X and tracks signal weight
        evolution, prediction MSE, and ENB.

Both return a BacktestResult dataclass with per-period arrays and summary
statistics.  The design is model-agnostic: any callable that implements
fit(X, y) / fit(returns) and exposes weights works as a drop-in.
"""

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from .portfolio_alloc import PortfolioAllocator
from .finance_alpha import AlphaSignalCombiner
from .utils_math import effective_number_of_bets, herfindahl_index


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Container for rolling backtest output.

    Attributes
    ----------
    weights         : (n_periods, n_assets) array of portfolio/signal weights
    dates           : (n_periods,) period end-dates (passed in by caller)
    realised_returns: (n_periods,) realised return for each period
    enb             : (n_periods,) effective number of bets
    hhi             : (n_periods,) Herfindahl index
    turnover        : (n_periods,) L1 weight change vs previous period
    pred_returns    : (n_periods,) predicted returns (signal backtest only)
    mse_per_period  : (n_periods,) OOS MSE per rebalancing step
    """
    weights:          np.ndarray
    dates:            np.ndarray = field(default_factory=lambda: np.array([]))
    realised_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    net_returns:      np.ndarray = field(default_factory=lambda: np.array([]))
    cost_drag:        np.ndarray = field(default_factory=lambda: np.array([]))
    enb:              np.ndarray = field(default_factory=lambda: np.array([]))
    hhi:              np.ndarray = field(default_factory=lambda: np.array([]))
    turnover:         np.ndarray = field(default_factory=lambda: np.array([]))
    pred_returns:     np.ndarray = field(default_factory=lambda: np.array([]))
    mse_per_period:   np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> dict:
        """Scalar summary statistics (gross and net of transaction costs)."""
        stats = {
            "n_periods":     len(self.enb),
            "mean_enb":      float(np.mean(self.enb))      if len(self.enb)      else np.nan,
            "mean_hhi":      float(np.mean(self.hhi))      if len(self.hhi)      else np.nan,
            "mean_turnover": float(np.mean(self.turnover))  if len(self.turnover) else np.nan,
        }
        if len(self.realised_returns):
            r = self.realised_returns
            stats.update({
                "total_return":  float(np.sum(r)),
                "mean_return":   float(np.mean(r)),
                "volatility":    float(np.std(r)),
                "sharpe":        float(np.mean(r) / (np.std(r) + 1e-12)),
                "max_drawdown":  float(_max_drawdown(r)),
            })
        if len(self.net_returns):
            rn = self.net_returns
            stats.update({
                "net_total_return": float(np.sum(rn)),
                "net_sharpe":       float(np.mean(rn) / (np.std(rn) + 1e-12)),
                "net_max_drawdown": float(_max_drawdown(rn)),
                "mean_cost_drag":   float(np.mean(self.cost_drag))
                                    if len(self.cost_drag) else np.nan,
                "total_cost_drag":  float(np.sum(self.cost_drag))
                                    if len(self.cost_drag) else np.nan,
            })
        if len(self.mse_per_period):
            stats["mean_oos_mse"] = float(np.mean(self.mse_per_period))
        return stats

    def print_summary(self) -> None:
        s = self.summary()
        print(f"  Periods:        {s['n_periods']}")
        print(f"  Mean ENB:       {s.get('mean_enb', float('nan')):.2f}")
        print(f"  Mean HHI:       {s.get('mean_hhi', float('nan')):.4f}")
        print(f"  Mean turnover:  {s.get('mean_turnover', float('nan')):.4f}")
        if "sharpe" in s:
            print(f"  --- Gross returns ---")
            print(f"  Total return:   {s['total_return']:.4f}")
            print(f"  Sharpe ratio:   {s['sharpe']:.3f}")
            print(f"  Max drawdown:   {s['max_drawdown']:.4f}")
        if "net_sharpe" in s:
            print(f"  --- Net of costs ---")
            print(f"  Net total ret:  {s['net_total_return']:.4f}")
            print(f"  Net Sharpe:     {s['net_sharpe']:.3f}")
            print(f"  Net drawdown:   {s['net_max_drawdown']:.4f}")
            print(f"  Total cost:     {s['total_cost_drag']:.6f}")
        if "mean_oos_mse" in s:
            print(f"  Mean OOS MSE:   {s['mean_oos_mse']:.6f}")


def _max_drawdown(returns: np.ndarray) -> float:
    cum = np.cumprod(1 + np.clip(returns, -0.9, None))
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / (running_max + 1e-12)
    return float(np.min(drawdowns))


# ---------------------------------------------------------------------------
# Rolling portfolio backtest
# ---------------------------------------------------------------------------

class RollingPortfolioBacktest:
    """
    Fit a PortfolioAllocator on a rolling window of returns data.

    Parameters
    ----------
    window : int
        Number of observations in each training window.
    refit_freq : int
        Re-fit every this many periods (1 = daily re-fit).
    lam : float
        Entropy regularisation strength passed to PortfolioAllocator.
    optimizer : str
        Optimiser passed to PortfolioAllocator.
    n_iters : int
        Max iterations per fit.
    learning_rate : float
        Step size passed to PortfolioAllocator.
    """

    def __init__(
        self,
        window:        int   = 252,
        refit_freq:    int   = 21,
        lam:           float = 0.05,
        optimizer:     str   = "mirror_descent",
        n_iters:       int   = 400,
        learning_rate: float = 0.1,
        cost_bps:      float = 0.0,
    ):
        self.window        = window
        self.refit_freq    = refit_freq
        self.lam           = lam
        self.optimizer     = optimizer
        self.n_iters       = n_iters
        self.learning_rate = learning_rate
        self.cost_bps      = cost_bps   # transaction cost in basis points per unit L1 turnover

    def run(
        self,
        returns: np.ndarray,
        dates:   Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run the rolling backtest.

        Parameters
        ----------
        returns : (T, N) array of period returns (T periods, N assets).
        dates   : (T,) optional array of dates / labels.

        Returns
        -------
        BacktestResult
        """
        T, N = returns.shape
        if T <= self.window + self.refit_freq:
            raise ValueError(
                f"Not enough data: T={T} <= window={self.window} + refit_freq={self.refit_freq}"
            )

        weights_list, real_ret_list, net_ret_list, cost_list = [], [], [], []
        enb_list, hhi_list, turn_list = [], [], []
        date_list = []
        prev_w = np.ones(N) / N

        for start in range(0, T - self.window - self.refit_freq + 1, self.refit_freq):
            end = start + self.window
            train_returns = returns[start:end]

            alloc = PortfolioAllocator(
                learning_rate=self.learning_rate,
                n_iters=self.n_iters,
                lam=self.lam,
                optimizer=self.optimizer,
            )
            alloc.fit(train_returns)
            w = alloc.get_weights()

            # Realised return over the next refit_freq periods
            oos_ret = returns[end:end + self.refit_freq]
            realised = float(np.sum(oos_ret @ w))

            # Transaction cost: cost_bps basis points per unit L1 turnover
            l1_turn = float(np.sum(np.abs(w - prev_w)))
            cost = self.cost_bps * 1e-4 * l1_turn
            net = realised - cost

            weights_list.append(w)
            real_ret_list.append(realised)
            net_ret_list.append(net)
            cost_list.append(cost)
            enb_list.append(effective_number_of_bets(w))
            hhi_list.append(herfindahl_index(w))
            turn_list.append(l1_turn)
            if dates is not None:
                date_list.append(dates[end - 1])
            prev_w = w.copy()

        return BacktestResult(
            weights=np.array(weights_list),
            dates=np.array(date_list) if date_list else np.arange(len(weights_list)),
            realised_returns=np.array(real_ret_list),
            net_returns=np.array(net_ret_list),
            cost_drag=np.array(cost_list),
            enb=np.array(enb_list),
            hhi=np.array(hhi_list),
            turnover=np.array(turn_list),
        )


# ---------------------------------------------------------------------------
# Rolling signal backtest
# ---------------------------------------------------------------------------

class RollingSignalBacktest:
    """
    Fit an AlphaSignalCombiner on a rolling window of (X, y) signal data.

    Parameters
    ----------
    window     : int   Training window length.
    refit_freq : int   Re-fit frequency.
    lam        : float Entropy regularisation.
    optimizer  : str   Optimiser.
    n_iters    : int   Max iterations.
    learning_rate : float Step size.
    """

    def __init__(
        self,
        window:        int   = 252,
        refit_freq:    int   = 21,
        lam:           float = 0.05,
        optimizer:     str   = "mirror_descent",
        n_iters:       int   = 400,
        learning_rate: float = 0.1,
    ):
        self.window        = window
        self.refit_freq    = refit_freq
        self.lam           = lam
        self.optimizer     = optimizer
        self.n_iters       = n_iters
        self.learning_rate = learning_rate

    def run(
        self,
        X:     np.ndarray,
        y:     np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run the rolling signal backtest.

        Parameters
        ----------
        X : (T, N) signal matrix.
        y : (T,) target returns.
        dates : optional (T,) date labels.

        Returns
        -------
        BacktestResult
        """
        from sklearn.metrics import mean_squared_error

        T, N = X.shape
        if T <= self.window + self.refit_freq:
            raise ValueError(f"Not enough data: T={T}")

        weights_list, pred_list, mse_list = [], [], []
        enb_list, hhi_list, turn_list = [], [], []
        date_list = []
        prev_w = np.ones(N) / N

        for start in range(0, T - self.window - self.refit_freq + 1, self.refit_freq):
            end = start + self.window
            combiner = AlphaSignalCombiner(
                learning_rate=self.learning_rate,
                n_iters=self.n_iters,
                lam=self.lam,
                optimizer=self.optimizer,
            )
            combiner.fit(X[start:end], y[start:end])
            w = combiner.get_signal_weights()

            # OOS prediction
            X_oos = X[end:end + self.refit_freq]
            y_oos = y[end:end + self.refit_freq]
            y_hat = combiner.predict(X_oos)
            oos_mse = float(mean_squared_error(y_oos, y_hat))

            weights_list.append(w)
            pred_list.extend(y_hat.tolist())
            mse_list.append(oos_mse)
            enb_list.append(effective_number_of_bets(w))
            hhi_list.append(herfindahl_index(w))
            turn_list.append(float(np.sum(np.abs(w - prev_w))))
            if dates is not None:
                date_list.append(dates[end - 1])
            prev_w = w.copy()

        return BacktestResult(
            weights=np.array(weights_list),
            dates=np.array(date_list) if date_list else np.arange(len(weights_list)),
            pred_returns=np.array(pred_list),
            mse_per_period=np.array(mse_list),
            enb=np.array(enb_list),
            hhi=np.array(hhi_list),
            turnover=np.array(turn_list),
        )
