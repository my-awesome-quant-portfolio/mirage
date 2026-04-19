"""
MIRAGE++ on real financial data — end-to-end demo.

Three experiments:
  1. Sector ETF portfolio weight estimation (11 ETFs -> SPY return)
  2. Fama-French 5-factor signal combination (FF5 -> stock return)
  3. Technical indicator ensemble (12 signals -> SPY 1-day return)

For each experiment:
  - Compare MIRAGE++ (4 optimisers) vs OLS / Ridge / Lasso / ElasticNet
  - 5-fold time-series cross-validation (no shuffling — preserves temporal order)
  - Report: test MSE, R², weight entropy, ENB, cosine distance, training time
  - Plot learned weight profiles

Run:
    python examples/market_experiments.py
    python examples/market_experiments.py --exp sector    # single experiment
    python examples/market_experiments.py --quick         # smaller date range
"""

import argparse
import copy
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from mirror_linear_regression import MirrorLinearRegression
from mirror_linear_regression.utils_math import (
    entropy, herfindahl_index, effective_number_of_bets,
)
from examples.market_datasets import (
    load_sector_etf_portfolio,
    load_fama_french_signals,
    load_technical_signals,
    describe,
)

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _simplex(coef):
    c = np.clip(np.asarray(coef, float).flatten(), 0, None)
    s = c.sum()
    return c / s if s > 1e-12 else np.ones(len(c)) / len(c)


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _build_models(iters=500):
    def mlr(opt, lam=0.01, lr=0.10):
        return MirrorLinearRegression(optimizer=opt, lam=lam,
                                      learning_rate=lr, n_iters=iters, tol=1e-7)
    return {
        "OLS":              ("sk",  LinearRegression(fit_intercept=False)),
        "Ridge(0.01)":      ("sk",  Ridge(alpha=0.01,  fit_intercept=False)),
        "Ridge(1.0)":       ("sk",  Ridge(alpha=1.0,   fit_intercept=False)),
        "Lasso(0.001)":     ("sk",  Lasso(alpha=0.001, fit_intercept=False, max_iter=5000)),
        "ElasticNet":       ("sk",  ElasticNet(alpha=0.01, l1_ratio=0.5,
                                               fit_intercept=False, max_iter=5000)),
        "MIRAGE-MD(0.01)":  ("mlr", mlr("mirror_descent",   0.01)),
        "MIRAGE-MD(0.05)":  ("mlr", mlr("mirror_descent",   0.05)),
        "MIRAGE-MD(0.1)":   ("mlr", mlr("mirror_descent",   0.10)),
        "MIRAGE-NGD":       ("mlr", mlr("natural_gradient",  0.01, lr=0.05)),
        "MIRAGE-MP":        ("mlr", mlr("mirror_prox",       0.01)),
        "MIRAGE-Ada":       ("mlr", mlr("ada_mirror",        0.01, lr=0.20)),
    }


def run_tscv(models, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}
    for name, (mtype, mobj) in models.items():
        mse_list, r2_list, t_list = [], [], []
        last_theta = None
        for tr, te in tscv.split(X):
            m = copy.deepcopy(mobj)
            t0 = time.perf_counter()
            m.fit(X[tr], y[tr])
            elapsed = time.perf_counter() - t0
            if mtype == "sk":
                theta = _simplex(m.coef_)
                yhat  = X[te] @ theta
            else:
                theta = m.weights
                yhat  = m.predict(X[te])
            mse_list.append(mean_squared_error(y[te], yhat))
            r2_list.append(r2_score(y[te], yhat))
            t_list.append(elapsed)
            last_theta = theta
        results[name] = {
            "mse":       float(np.mean(mse_list)),
            "mse_std":   float(np.std(mse_list)),
            "r2":        float(np.mean(r2_list)),
            "time":      float(np.mean(t_list)),
            "theta":     last_theta,
            "H":         float(entropy(last_theta)),
            "ENB":       float(effective_number_of_bets(last_theta)),
            "HHI":       float(herfindahl_index(last_theta)),
        }
    return results


def print_table(results, feature_names=None):
    hdr = (f"{'Model':<22} {'TestMSE':>11} {'+-':>8} {'R2':>8}"
           f" {'H':>6} {'ENB':>6} {'HHI':>7} {'t(s)':>7}")
    print(hdr)
    print("-" * len(hdr))
    prev_sklearn = True
    for name, r in results.items():
        is_mirage = name.startswith("MIRAGE")
        if is_mirage and prev_sklearn:
            print("-" * len(hdr))
            prev_sklearn = False
        print(f"{name:<22} {r['mse']:>11.6f} {r['mse_std']:>8.6f}"
              f" {r['r2']:>8.4f} {r['H']:>6.3f} {r['ENB']:>6.1f}"
              f" {r['HHI']:>7.4f} {r['time']:>7.3f}")


def plot_weights(results, feature_names, title, filename, highlight=None):
    models_to_plot = {k: v for k, v in results.items()
                      if k in ("OLS", "Ridge(1.0)", "MIRAGE-MD(0.01)",
                               "MIRAGE-MD(0.1)", "MIRAGE-Ada")}
    n = len(models_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, models_to_plot.items()):
        theta = r["theta"]
        colors = ["steelblue"] * len(theta)
        bars = ax.bar(range(len(theta)), theta, color=colors, alpha=0.8)
        ax.set_xticks(range(len(theta)))
        ax.set_xticklabels(feature_names[:len(theta)], rotation=45,
                           ha="right", fontsize=7)
        ax.set_title(f"{name}\nH={r['H']:.2f}  ENB={r['ENB']:.1f}", fontsize=9)
        ax.set_ylabel("weight")
    fig.suptitle(title, fontweight="bold", fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Experiment 1: Sector ETF portfolio
# ---------------------------------------------------------------------------

def exp_sector(quick=False):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: S&P 500 Sector ETF Portfolio")
    print("=" * 70)
    start = "2019-01-01" if quick else "2015-01-01"
    data = load_sector_etf_portfolio(start=start, end="2024-01-01")
    describe(data)

    X, y = data["X"], data["y"]
    models = _build_models(iters=300 if quick else 600)
    print(f"\nRunning {len(models)} models x 5-fold time-series CV...")
    results = run_tscv(models, X, y)
    print_table(results, data["sector_names"])
    plot_weights(results, data["sector_names"],
                 "Sector ETF portfolio weights (11 sectors)",
                 "sector_etf_weights.png")

    # Lambda sensitivity on this task
    print("\n  Lambda sensitivity (MIRAGE-MD):")
    lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    print(f"  {'lam':>8} {'CV_MSE':>10} {'H':>8} {'ENB':>8}")
    for lam in lambdas:
        m = MirrorLinearRegression(lam=lam, n_iters=400, tol=1e-7)
        r = run_tscv({"m": ("mlr", m)}, X, y)["m"]
        print(f"  {lam:>8.3f} {r['mse']:>10.6f} {r['H']:>8.3f} {r['ENB']:>8.1f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Fama-French factor signal combination
# ---------------------------------------------------------------------------

def exp_fama_french(quick=False):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Fama-French 5-Factor Signal Combination")
    print("=" * 70)
    start = "2015-01-01" if quick else "2010-01-01"
    data = load_fama_french_signals(target_ticker="SPY", start=start, end="2024-01-01")
    describe(data)

    X, y = data["X"], data["y"]

    # For monthly data we use 3 splits (fewer samples)
    models = _build_models(iters=200 if quick else 400)
    n_splits = 3 if quick else 5
    print(f"\nRunning {len(models)} models x {n_splits}-fold time-series CV...")
    results = run_tscv(models, X, y, n_splits=n_splits)
    print_table(results, data["factor_names"])
    plot_weights(results, data["factor_names"],
                 "FF5 factor loadings (5 factors -> SPY excess return)",
                 "ff5_factor_weights.png")

    print("\n  Diversification: how evenly does MIRAGE distribute factor exposure?")
    print(f"  Max-ENB (uniform) = {len(data['factor_names']):.1f}")
    for name in ["OLS", "MIRAGE-MD(0.01)", "MIRAGE-MD(0.1)"]:
        if name in results:
            r = results[name]
            print(f"  {name:<22} ENB={r['ENB']:.2f}  theta={r['theta'].round(3)}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Technical indicator ensemble
# ---------------------------------------------------------------------------

def exp_technical(quick=False):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Technical Indicator Ensemble (12 signals -> SPY)")
    print("=" * 70)
    start = "2015-01-01" if quick else "2010-01-01"
    data = load_technical_signals(ticker="SPY", start=start, end="2024-01-01")
    describe(data)

    X, y = data["X"], data["y"]
    models = _build_models(iters=300 if quick else 600)
    print(f"\nRunning {len(models)} models x 5-fold time-series CV...")
    results = run_tscv(models, X, y)
    print_table(results, data["signal_names"])
    plot_weights(results, data["signal_names"],
                 "Technical signal weights (12 signals -> SPY next-day return)",
                 "technical_signal_weights.png")

    # Rolling window: track how weights evolve over time
    print("\n  Rolling window weight stability (MIRAGE-MD, window=252)...")
    window = 252
    step   = 21
    m_roll = MirrorLinearRegression(lam=0.01, n_iters=300, tol=1e-7)
    weight_history = []
    date_history   = []
    for start_idx in range(0, len(X) - window, step):
        end_idx = start_idx + window
        m_roll_copy = copy.deepcopy(m_roll)
        m_roll_copy.fit(X[start_idx:end_idx], y[start_idx:end_idx])
        weight_history.append(m_roll_copy.weights)
        date_history.append(data["dates"][end_idx - 1])
    W = np.array(weight_history)
    enb_over_time = [effective_number_of_bets(w) for w in W]
    print(f"  Mean ENB over time: {np.mean(enb_over_time):.2f}"
          f"  ± {np.std(enb_over_time):.2f}")
    print(f"  Min ENB: {np.min(enb_over_time):.2f}  Max ENB: {np.max(enb_over_time):.2f}")

    # Save rolling weight figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for i, name in enumerate(data["signal_names"]):
        ax1.plot(range(len(W)), W[:, i], alpha=0.6, lw=1, label=name)
    ax1.set_ylabel("weight"); ax1.set_title("Rolling 252-day weights (MIRAGE-MD)")
    ax1.legend(fontsize=6, ncol=4)
    ax2.plot(range(len(enb_over_time)), enb_over_time, color="navy", lw=1.5)
    ax2.axhline(np.mean(enb_over_time), color="red", ls="--", lw=1, label="mean ENB")
    ax2.set_xlabel("rebalancing step"); ax2.set_ylabel("ENB")
    ax2.set_title("Effective Number of Bets over time"); ax2.legend()
    plt.tight_layout()
    out = FIG_DIR / "rolling_weights_technical.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIRAGE++ real data demo")
    parser.add_argument("--exp",   nargs="*", default=["sector", "ff", "tech"],
                        help="Experiments to run: sector ff tech")
    parser.add_argument("--quick", action="store_true",
                        help="Shorter date range, fewer iterations")
    args = parser.parse_args()

    chosen = {e.lower() for e in args.exp}
    all_results = {}

    if "sector" in chosen:
        all_results["sector"] = exp_sector(quick=args.quick)
    if any(k in chosen for k in ("ff", "fama", "fama_french")):
        all_results["ff"] = exp_fama_french(quick=args.quick)
    if any(k in chosen for k in ("tech", "technical")):
        all_results["tech"] = exp_technical(quick=args.quick)

    print("\n" + "=" * 70)
    print("DONE. Figures saved to:", FIG_DIR)
    print("=" * 70)
