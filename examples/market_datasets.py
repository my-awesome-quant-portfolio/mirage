"""
Real financial dataset loaders for MIRAGE++ experiments.

Three data sources, all fetched via yfinance / pandas-datareader and
cached to data/ so repeated runs are instant.

Functions
---------
load_sector_etf_portfolio
    Daily returns for 11 S&P 500 sector ETFs (2015-2024).
    Use case: simplex portfolio weight estimation.

load_fama_french_signals
    Fama-French 5-factor returns + a target equity/fund return.
    Use case: factor-exposure estimation (sum-to-1 constraint).

load_technical_signals
    12 momentum / vol / mean-reversion signals computed from
    a single ticker's OHLCV history.
    Use case: alpha signal combination (weights sum to 1).

load_equity_universe
    Daily returns for a user-specified list of tickers.
    Use case: custom portfolio, stress tests, backtesting.

Each function returns NumPy arrays so they plug directly into
MirrorLinearRegression with no extra pre-processing.
"""

import hashlib
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> pathlib.Path:
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return DATA_DIR / f"{key}_{h}.parquet"


def _yf_download(tickers, start, end) -> pd.DataFrame:
    """Download adjusted close prices; return daily log-returns."""
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.ffill().dropna(how="all")
    returns = np.log(close / close.shift(1)).iloc[1:]
    return returns.dropna(how="all").fillna(0.0)


def _load_or_fetch(cache_key: str, fetcher):
    p = _cache_path(cache_key)
    if p.exists():
        return pd.read_parquet(p)
    df = fetcher()
    df.to_parquet(p)
    return df


# ---------------------------------------------------------------------------
# 1. Sector ETF portfolio
# ---------------------------------------------------------------------------

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLY":  "Consumer Discr",
    "XLP":  "Consumer Staples",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLC":  "Comm Services",
}


def load_sector_etf_portfolio(
    start: str = "2015-01-01",
    end:   str = "2024-01-01",
    target: str = "SPY",
    cache: bool = True,
) -> dict:
    """
    Download daily log-returns for the 11 SPDR sector ETFs plus a target.

    The regression task is:
        y  = target daily log-return
        X  = lagged sector-ETF returns (1-day lag to avoid look-ahead bias)

    Returns a dict with keys:
        X          : (T, 11) feature matrix (lagged returns)
        y          : (T,)    target returns
        returns_df : full returns DataFrame (T x 12) with no lag
        tickers    : list of ETF tickers
        sector_names : list of sector names
        dates      : DatetimeIndex
    """
    tickers = list(SECTOR_ETFS.keys()) + [target]
    key = f"sector_etf_{start}_{end}_{'_'.join(sorted(tickers))}"

    def fetch():
        return _yf_download(tickers, start, end)

    returns_df = _load_or_fetch(key, fetch) if cache else _yf_download(tickers, start, end)

    # Drop rows where more than 2 assets are missing
    returns_df = returns_df.dropna(thresh=len(tickers) - 2).fillna(0.0)

    etf_cols = [t for t in list(SECTOR_ETFS.keys()) if t in returns_df.columns]
    target_col = target if target in returns_df.columns else returns_df.columns[-1]

    # 1-day lagged X to avoid look-ahead bias
    X = returns_df[etf_cols].shift(1).iloc[1:].values.astype(np.float64)
    y = returns_df[target_col].iloc[1:].values.astype(np.float64)
    dates = returns_df.index[1:]

    return {
        "X":            X,
        "y":            y,
        "returns_df":   returns_df,
        "tickers":      etf_cols,
        "sector_names": [SECTOR_ETFS[t] for t in etf_cols],
        "dates":        dates,
        "n_samples":    len(y),
        "n_features":   len(etf_cols),
        "description":  f"Sector ETF portfolio: {len(etf_cols)} ETFs, "
                        f"{len(y)} daily obs ({start} - {end})",
    }


# ---------------------------------------------------------------------------
# 2. Fama-French 5-factor signal combination
# ---------------------------------------------------------------------------

def load_fama_french_signals(
    target_ticker: str = "SPY",
    start: str = "2010-01-01",
    end:   str = "2024-01-01",
    n_factors: int = 5,
    cache: bool = True,
) -> dict:
    """
    Fama-French 5-factor model: use FF factors as alpha signals to predict
    a target asset's excess return.

    Factors (n_factors=5): Mkt-RF, SMB, HML, RMW, CMA
    Factors (n_factors=3): Mkt-RF, SMB, HML

    Task:
        y  = target monthly excess return (after subtracting RF)
        X  = n_factors factor returns  [each column sums-to-1 not needed here;
             MIRAGE++ learns factor *loadings* on the simplex]

    Returns dict with X, y, factor_names, dates, description.
    """
    import pandas_datareader.data as web

    ff_name = "F-F_Research_Data_5_Factors_2x3" if n_factors == 5 \
              else "F-F_Research_Data_Factors"
    key = f"ff{n_factors}_{target_ticker}_{start}_{end}"

    def fetch_ff():
        ff = web.DataReader(ff_name, "famafrench", start=start, end=end)[0]
        ff = ff / 100.0
        return ff

    ff_df = _load_or_fetch(f"ff_{ff_name}_{start}_{end}", fetch_ff) if cache \
            else fetch_ff()

    # Download target monthly returns
    import yfinance as yf
    raw = yf.download(target_ticker, start=start, end=end,
                      auto_adjust=True, progress=False, interval="1mo")
    close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    monthly_ret = np.log(close / close.shift(1)).dropna()
    monthly_ret.index = monthly_ret.index.to_period("M")

    if not isinstance(ff_df.index, pd.PeriodIndex):
        ff_df.index = ff_df.index.to_period("M")
    common_idx = ff_df.index.intersection(monthly_ret.index)
    ff_aligned = ff_df.loc[common_idx]
    ret_aligned = monthly_ret.loc[common_idx]

    factor_cols = [c for c in ff_aligned.columns if c != "RF"][:n_factors]
    X = ff_aligned[factor_cols].values.astype(np.float64)

    # Excess return
    rf = ff_aligned["RF"].values if "RF" in ff_aligned.columns else 0.0
    y  = (ret_aligned.values - rf).astype(np.float64)
    dates = pd.period_range(start=common_idx[0], end=common_idx[-1], freq="M")

    return {
        "X":             X,
        "y":             y,
        "factor_names":  factor_cols,
        "dates":         dates,
        "n_samples":     len(y),
        "n_features":    n_factors,
        "description":   f"FF{n_factors} factors -> {target_ticker} excess return: "
                         f"{len(y)} monthly obs ({start} - {end})",
    }


# ---------------------------------------------------------------------------
# 3. Technical indicator ensemble
# ---------------------------------------------------------------------------

def _momentum(prices: pd.Series, window: int) -> pd.Series:
    return prices.pct_change(window)

def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mu  = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mu) / (std + 1e-9)

def _vol_ratio(prices: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    rv_short = prices.pct_change().rolling(short).std()
    rv_long  = prices.pct_change().rolling(long).std()
    return rv_short / (rv_long + 1e-9)


def load_technical_signals(
    ticker: str = "SPY",
    start:  str = "2010-01-01",
    end:    str = "2024-01-01",
    forward_days: int = 1,
    cache: bool = True,
) -> dict:
    """
    Build a matrix of 12 technical alpha signals from a single asset's history.

    Signals (all normalized via rolling z-score):
        mom_5, mom_21, mom_63, mom_126   -- price momentum (5/21/63/126 days)
        rsi_14                           -- 14-day RSI
        ma_cross_20_50                   -- MA(20)/MA(50) cross signal
        ma_cross_50_200                  -- MA(50)/MA(200) cross signal
        vol_ratio_5_20                   -- realised vol ratio (short/long)
        vol_ratio_10_60                  -- realised vol ratio
        reversal_5                       -- short-term mean reversion
        price_vs_52w_high                -- drawdown from 52-week high
        price_vs_52w_low                 -- distance from 52-week low

    Target: forward_days-ahead log return.

    Returns dict with X, y, signal_names, dates, description.
    """
    key = f"tech_{ticker}_{start}_{end}"

    def fetch():
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        # yfinance may return DataFrame with MultiIndex columns for single ticker
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close

    price_file = _cache_path(key + "_price")
    if cache and price_file.exists():
        prices = pd.read_parquet(price_file).iloc[:, 0]
    else:
        prices = fetch()
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name="close")
        else:
            prices = pd.DataFrame({"close": prices})
        prices.to_parquet(price_file)
        prices = prices.iloc[:, 0]

    signals = pd.DataFrame(index=prices.index)
    signals["mom_5"]          = _zscore(_momentum(prices, 5))
    signals["mom_21"]         = _zscore(_momentum(prices, 21))
    signals["mom_63"]         = _zscore(_momentum(prices, 63))
    signals["mom_126"]        = _zscore(_momentum(prices, 126))
    signals["rsi_14"]         = _zscore(_rsi(prices, 14))
    signals["ma_cross_20_50"] = _zscore(
        prices.rolling(20).mean() / (prices.rolling(50).mean() + 1e-9) - 1)
    signals["ma_cross_50_200"] = _zscore(
        prices.rolling(50).mean() / (prices.rolling(200).mean() + 1e-9) - 1)
    signals["vol_ratio_5_20"]  = _zscore(_vol_ratio(prices, 5,  20))
    signals["vol_ratio_10_60"] = _zscore(_vol_ratio(prices, 10, 60))
    signals["reversal_5"]      = _zscore(-_momentum(prices, 5))
    signals["vs_52w_high"]     = _zscore(
        prices / (prices.rolling(252).max() + 1e-9) - 1)
    signals["vs_52w_low"]      = _zscore(
        prices / (prices.rolling(252).min() + 1e-9) - 1)

    # Forward return (target)
    fwd_ret = np.log(prices.shift(-forward_days) / prices)

    combined = signals.copy()
    combined["fwd_ret"] = fwd_ret
    combined = combined.dropna()

    X = combined[list(signals.columns)].values.astype(np.float64)
    y = combined["fwd_ret"].values.astype(np.float64)

    return {
        "X":             X,
        "y":             y,
        "signal_names":  list(signals.columns),
        "dates":         combined.index,
        "n_samples":     len(y),
        "n_features":    X.shape[1],
        "description":   f"Technical signals for {ticker}: "
                         f"{X.shape[1]} signals, {len(y)} obs ({start} - {end})",
    }


# ---------------------------------------------------------------------------
# 4. Generic equity universe
# ---------------------------------------------------------------------------

_SP100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK-B", "JPM",
    "JNJ",  "V",    "UNH",  "XOM",  "PG",   "MA",    "HD",   "CVX",   "MRK",
    "ABBV", "COST", "PEP",  "KO",   "WMT",  "BAC",   "CRM",  "ACN",   "MCD",
    "LIN",  "TMO",  "CSCO", "ABT",  "AVGO", "WFC",   "DHR",  "NKE",   "TXN",
    "NEE",  "PM",   "RTX",  "HON",  "IBM",  "AMGN",  "MDT",  "ORCL",  "GE",
    "QCOM", "UPS",  "LOW",  "INTC", "CAT",
]


def load_equity_universe(
    tickers: Optional[list] = None,
    start: str = "2018-01-01",
    end:   str = "2024-01-01",
    target: str = "SPY",
    min_history_frac: float = 0.9,
    cache: bool = True,
) -> dict:
    """
    Download daily returns for a list of equities.

    Filters out tickers with fewer than min_history_frac * T observations.

    Returns dict with:
        X      : (T, N) returns matrix (lagged 1 day)
        y      : (T,)   target daily return
        returns_df : full unlagged returns DataFrame
        tickers : list of surviving tickers
        dates   : DatetimeIndex
    """
    if tickers is None:
        tickers = _SP100[:30]
    all_tickers = sorted(set(tickers) | {target})
    key = f"equity_{'_'.join(sorted(all_tickers))}_{start}_{end}"

    def fetch():
        return _yf_download(all_tickers, start, end)

    returns_df = _load_or_fetch(key, fetch) if cache else _yf_download(all_tickers, start, end)

    # Filter tickers with insufficient history
    T = len(returns_df)
    good = [c for c in returns_df.columns
            if returns_df[c].notna().sum() >= min_history_frac * T]
    returns_df = returns_df[good].fillna(0.0)

    feature_cols = [c for c in returns_df.columns if c != target]
    target_col   = target if target in returns_df.columns else returns_df.columns[-1]

    X = returns_df[feature_cols].shift(1).iloc[1:].values.astype(np.float64)
    y = returns_df[target_col].iloc[1:].values.astype(np.float64)
    dates = returns_df.index[1:]

    return {
        "X":          X,
        "y":          y,
        "returns_df": returns_df,
        "tickers":    feature_cols,
        "dates":      dates,
        "n_samples":  len(y),
        "n_features": len(feature_cols),
        "description": f"Equity universe: {len(feature_cols)} stocks, "
                       f"{len(y)} daily obs ({start} - {end})",
    }


# ---------------------------------------------------------------------------
# Quick summary
# ---------------------------------------------------------------------------

def describe(data: dict) -> None:
    print(data["description"])
    print(f"  X shape : {data['X'].shape}")
    print(f"  y shape : {data['y'].shape}")
    print(f"  y range : [{data['y'].min():.4f}, {data['y'].max():.4f}]")
    print(f"  X range : [{data['X'].min():.4f}, {data['X'].max():.4f}]")
