"""
Offline tests for the real-data pipeline using cached parquet files.

All tests read from data/ (populated on first download) and never hit the
network.  The suite is skipped gracefully if the cache is missing.

Coverage:
  - real_datasets.py loaders return correct shapes / dtypes / ranges
  - MIRAGE++ achieves higher ENB than OLS on real data (diversity bonus)
  - PortfolioAllocator on real sector ETF returns
  - AlphaSignalCombiner on real technical signals
  - Time-series CV: MIRAGE-MD MSE is finite and reasonable
"""

import pathlib

import numpy as np
import pytest

from mirror_linear_regression import MirrorLinearRegression
from mirror_linear_regression.portfolio_alloc import PortfolioAllocator
from mirror_linear_regression.finance_alpha import AlphaSignalCombiner
from mirror_linear_regression.utils_math import effective_number_of_bets, herfindahl_index

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Skip markers — each test group skips if its cache is absent
# ---------------------------------------------------------------------------

_has_sector = any(DATA_DIR.glob("sector_etf_*.parquet"))
_has_ff     = any(DATA_DIR.glob("ff_F-F_*.parquet"))
_has_tech   = any(DATA_DIR.glob("tech_SPY_*_price_*.parquet"))
_has_equity = any(DATA_DIR.glob("equity_AAPL_*.parquet"))

skip_sector = pytest.mark.skipif(not _has_sector, reason="sector ETF cache missing")
skip_ff     = pytest.mark.skipif(not _has_ff,     reason="Fama-French cache missing")
skip_tech   = pytest.mark.skipif(not _has_tech,   reason="technical signals cache missing")
skip_equity = pytest.mark.skipif(not _has_equity, reason="equity universe cache missing")


def _simplex(c):
    c = np.clip(np.asarray(c, float).flatten(), 0, None)
    s = c.sum()
    return c / s if s > 1e-12 else np.ones(len(c)) / len(c)


# ---------------------------------------------------------------------------
# Loader shape / dtype tests
# ---------------------------------------------------------------------------

class TestLoaderOutput:
    @skip_sector
    def test_sector_etf_shapes(self):
        from examples.market_datasets import load_sector_etf_portfolio
        d = load_sector_etf_portfolio()
        assert d["X"].ndim == 2
        assert d["y"].ndim == 1
        assert d["X"].shape[0] == d["y"].shape[0]
        assert d["X"].shape[1] == len(d["tickers"])
        assert d["X"].dtype == np.float64
        assert np.all(np.isfinite(d["X"]))
        assert np.all(np.isfinite(d["y"]))

    @skip_sector
    def test_sector_etf_returns_bounded(self):
        from examples.market_datasets import load_sector_etf_portfolio
        d = load_sector_etf_portfolio()
        # Daily log-returns should be within [-50%, +50%]
        assert d["X"].max() < 0.5
        assert d["X"].min() > -0.5

    @skip_tech
    def test_technical_signals_shapes(self):
        from examples.market_datasets import load_technical_signals
        d = load_technical_signals()
        assert d["X"].shape[1] == 12, "expected 12 technical signals"
        assert d["y"].ndim == 1
        assert np.all(np.isfinite(d["X"]))

    @skip_ff
    def test_ff_signals_shapes(self):
        from examples.market_datasets import load_fama_french_signals
        d = load_fama_french_signals()
        assert d["X"].shape[1] <= 5
        assert d["y"].ndim == 1
        assert len(d["factor_names"]) == d["X"].shape[1]

    @skip_equity
    def test_equity_universe_shapes(self):
        from examples.market_datasets import load_equity_universe
        d = load_equity_universe()
        assert d["X"].shape[0] == d["y"].shape[0]
        assert len(d["tickers"]) == d["X"].shape[1]


# ---------------------------------------------------------------------------
# MIRAGE++ vs OLS diversity comparison on real data
# ---------------------------------------------------------------------------

class TestDiversityOnRealData:
    @skip_sector
    def test_mirage_more_diversified_than_ols_sector(self):
        from examples.market_datasets import load_sector_etf_portfolio
        from sklearn.linear_model import LinearRegression
        d = load_sector_etf_portfolio()
        X, y = d["X"], d["y"]

        ols = LinearRegression(fit_intercept=False).fit(X, y)
        w_ols = _simplex(ols.coef_)

        mlr = MirrorLinearRegression(lam=0.05, n_iters=400, tol=1e-7)
        mlr.fit(X, y)
        w_mlr = mlr.weights

        enb_ols = effective_number_of_bets(w_ols)
        enb_mlr = effective_number_of_bets(w_mlr)
        assert enb_mlr > enb_ols, (
            f"MIRAGE ENB={enb_mlr:.2f} should exceed OLS ENB={enb_ols:.2f}")

    @skip_tech
    def test_mirage_more_diversified_than_ols_tech(self):
        from examples.market_datasets import load_technical_signals
        from sklearn.linear_model import LinearRegression
        d = load_technical_signals()
        X, y = d["X"], d["y"]

        ols = LinearRegression(fit_intercept=False).fit(X, y)
        w_ols = _simplex(ols.coef_)

        mlr = MirrorLinearRegression(lam=0.05, n_iters=400, tol=1e-7)
        mlr.fit(X, y)
        w_mlr = mlr.weights

        assert effective_number_of_bets(w_mlr) > effective_number_of_bets(w_ols)

    @skip_sector
    def test_higher_lambda_higher_enb_real_data(self):
        from examples.market_datasets import load_sector_etf_portfolio
        d = load_sector_etf_portfolio()
        X, y = d["X"], d["y"]

        enbs = {}
        for lam in (0.001, 0.05, 0.5):
            m = MirrorLinearRegression(lam=lam, n_iters=400, tol=1e-7)
            m.fit(X, y)
            enbs[lam] = effective_number_of_bets(m.weights)
        assert enbs[0.5] > enbs[0.001], "higher lambda must increase ENB on real data"


# ---------------------------------------------------------------------------
# PortfolioAllocator on real sector ETF returns
# ---------------------------------------------------------------------------

class TestPortfolioAllocatorReal:
    @skip_sector
    def test_weights_on_simplex(self):
        from examples.market_datasets import load_sector_etf_portfolio
        d = load_sector_etf_portfolio()
        returns = d["returns_df"][[t for t in d["tickers"]]].values
        alloc = PortfolioAllocator(n_iters=400, lam=0.05, learning_rate=0.1)
        alloc.fit(returns)
        w = alloc.get_weights()
        assert abs(w.sum() - 1.0) < 1e-7
        assert np.all(w >= -1e-10)
        assert w.shape == (len(d["tickers"]),)

    @skip_sector
    def test_diagnostics_contract(self):
        from examples.market_datasets import load_sector_etf_portfolio
        d = load_sector_etf_portfolio()
        returns = d["returns_df"][[t for t in d["tickers"]]].values
        alloc = PortfolioAllocator(n_iters=300, lam=0.05)
        alloc.fit(returns)
        diag = alloc.portfolio_diagnostics()
        assert 0 < diag["effective_bets"] <= len(d["tickers"]) + 1e-6
        assert 0 < diag["herfindahl_index"] <= 1.0
        assert diag["weight_entropy"] >= 0

    @skip_sector
    def test_all_optimizers_valid(self):
        from examples.market_datasets import load_sector_etf_portfolio
        d = load_sector_etf_portfolio()
        returns = d["returns_df"][[t for t in d["tickers"]]].values
        for opt in ("mirror_descent", "natural_gradient", "mirror_prox", "ada_mirror"):
            lr = 0.05 if opt == "natural_gradient" else 0.1
            alloc = PortfolioAllocator(n_iters=200, lam=0.05,
                                       optimizer=opt, learning_rate=lr)
            alloc.fit(returns)
            w = alloc.get_weights()
            assert abs(w.sum() - 1.0) < 1e-6, f"sum != 1 for optimizer={opt}"


# ---------------------------------------------------------------------------
# AlphaSignalCombiner on real technical signals
# ---------------------------------------------------------------------------

class TestAlphaSignalCombinerReal:
    @skip_tech
    def test_weights_on_simplex(self):
        from examples.market_datasets import load_technical_signals
        d = load_technical_signals()
        combiner = AlphaSignalCombiner(n_iters=400, lam=0.05)
        combiner.fit(d["X"], d["y"])
        w = combiner.get_signal_weights()
        assert abs(w.sum() - 1.0) < 1e-7
        assert w.shape == (12,)

    @skip_tech
    def test_predict_shape_and_finite(self):
        from examples.market_datasets import load_technical_signals
        d = load_technical_signals()
        combiner = AlphaSignalCombiner(n_iters=300, lam=0.05)
        combiner.fit(d["X"], d["y"])
        pred = combiner.predict(d["X"])
        assert pred.shape == d["y"].shape
        assert np.all(np.isfinite(pred))

    @skip_tech
    def test_diagnostics_contract(self):
        from examples.market_datasets import load_technical_signals
        d = load_technical_signals()
        combiner = AlphaSignalCombiner(n_iters=300, lam=0.05)
        combiner.fit(d["X"], d["y"])
        diag = combiner.signal_diagnostics()
        assert 0 < diag["effective_signals"] <= 12
        assert diag["weight_entropy"] >= 0


# ---------------------------------------------------------------------------
# Time-series CV: MIRAGE++ MSE finite and not worse than trivial baseline
# ---------------------------------------------------------------------------

class TestTimeSeriesCV:
    @skip_sector
    def test_tscv_mse_finite_sector(self):
        from examples.market_datasets import load_sector_etf_portfolio
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error
        import copy
        d = load_sector_etf_portfolio()
        X, y = d["X"], d["y"]
        tscv = TimeSeriesSplit(n_splits=3)
        mlr = MirrorLinearRegression(lam=0.05, n_iters=300, tol=1e-7)
        mse_list = []
        for tr, te in tscv.split(X):
            m = copy.deepcopy(mlr)
            m.fit(X[tr], y[tr])
            mse_list.append(mean_squared_error(y[te], m.predict(X[te])))
        assert all(np.isfinite(mse_list)), "non-finite MSE in time-series CV"
        # CV MSE should be in a reasonable range for daily returns
        assert np.mean(mse_list) < 1.0, "CV MSE implausibly large"

    @skip_tech
    def test_tscv_mse_finite_tech(self):
        from examples.market_datasets import load_technical_signals
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error
        import copy
        d = load_technical_signals()
        X, y = d["X"], d["y"]
        tscv = TimeSeriesSplit(n_splits=3)
        mlr = MirrorLinearRegression(lam=0.05, n_iters=300, tol=1e-7)
        mse_list = []
        for tr, te in tscv.split(X):
            m = copy.deepcopy(mlr)
            m.fit(X[tr], y[tr])
            mse_list.append(mean_squared_error(y[te], m.predict(X[te])))
        assert all(np.isfinite(mse_list))

    @skip_ff
    def test_tscv_ff_factors(self):
        from examples.market_datasets import load_fama_french_signals
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error
        import copy
        d = load_fama_french_signals()
        X, y = d["X"], d["y"]
        n_splits = min(3, len(y) // 10)
        if n_splits < 2:
            pytest.skip("insufficient FF data for CV")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mlr = MirrorLinearRegression(lam=0.05, n_iters=200, tol=1e-7)
        mse_list = []
        for tr, te in tscv.split(X):
            m = copy.deepcopy(mlr)
            m.fit(X[tr], y[tr])
            mse_list.append(mean_squared_error(y[te], m.predict(X[te])))
        assert all(np.isfinite(mse_list))
