"""
benchmark.py — MIRAGE++ vs classical linear regression baselines.

Compares MIRAGE++ (4 optimisers) against OLS, Ridge, Lasso, and ElasticNet
on six quantitative finance scenarios using k-fold cross-validation.

Metrics reported per model per dataset:
  test_mse       Out-of-sample mean squared error (primary)
  train_mse      In-sample MSE
  test_r2        Out-of-sample R²
  weight_entropy Shannon entropy of learned weights H(θ)
  hhi            Herfindahl-Hirschman Index (weight concentration)
  enb            Effective Number of Bets = exp(H(θ))
  weight_cosine  Cosine similarity with ground-truth weights (where available)
  weight_l1      L1 distance from ground-truth weights (where available)
  time_s         Wall-clock training time (seconds)

Run:
    python benchmark.py
    python benchmark.py --dataset alpha     # single dataset
    python benchmark.py --quick             # fewer folds / iterations
"""

import sys
import time
import copy
import warnings
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

from mirror_linear_regression import MirrorLinearRegression
from mirror_linear_regression.bregman import ItakuraSaitoDivergence
from mirror_linear_regression.utils_math import (
    entropy, herfindahl_index, effective_number_of_bets,
)
from examples.synthetic_datasets import (
    toy_alpha_signal_combination,
    toy_sparse_portfolio,
    toy_volatility_forecasting,
    toy_ensemble_forecasting,
    toy_macro_predictive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_simplex(coef: np.ndarray) -> np.ndarray:
    """Project sklearn coefficients onto the probability simplex (clip + normalise)."""
    coef = np.clip(np.asarray(coef, dtype=float).flatten(), 0.0, None)
    total = coef.sum()
    return coef / total if total > 1e-12 else np.ones(len(coef)) / len(coef)


def _weight_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _weight_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _mirage(optimizer, lam, lr, n_iters, divergence=None):
    return MirrorLinearRegression(
        optimizer=optimizer, lam=lam, learning_rate=lr,
        n_iters=n_iters, tol=1e-7, divergence=divergence,
    )


def build_model_registry(quick: bool = False) -> dict:
    iters = 300 if quick else 600

    return {
        # --- Baselines (sklearn, projected to simplex post-hoc) ---
        "OLS":              ("sklearn", LinearRegression(fit_intercept=False)),
        "Ridge(α=0.001)":   ("sklearn", Ridge(alpha=0.001, fit_intercept=False)),
        "Ridge(α=0.1)":     ("sklearn", Ridge(alpha=0.1,   fit_intercept=False)),
        "Ridge(α=1.0)":     ("sklearn", Ridge(alpha=1.0,   fit_intercept=False)),
        "Lasso(α=0.001)":   ("sklearn", Lasso(alpha=0.001, fit_intercept=False, max_iter=5000)),
        "Lasso(α=0.01)":    ("sklearn", Lasso(alpha=0.01,  fit_intercept=False, max_iter=5000)),
        "ElasticNet":       ("sklearn", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                                   fit_intercept=False, max_iter=5000)),
        # --- MIRAGE++ variants ---
        "MIRAGE-MD(λ=0.001)":  ("mirage", _mirage("mirror_descent",   0.001, 0.10, iters)),
        "MIRAGE-MD(λ=0.01)":   ("mirage", _mirage("mirror_descent",   0.01,  0.10, iters)),
        "MIRAGE-MD(λ=0.05)":   ("mirage", _mirage("mirror_descent",   0.05,  0.10, iters)),
        "MIRAGE-MD(λ=0.1)":    ("mirage", _mirage("mirror_descent",   0.10,  0.10, iters)),
        "MIRAGE-NGD(λ=0.01)":  ("mirage", _mirage("natural_gradient", 0.01,  0.05, iters)),
        "MIRAGE-MP(λ=0.01)":   ("mirage", _mirage("mirror_prox",      0.01,  0.10, iters)),
        "MIRAGE-Ada(λ=0.01)":  ("mirage", _mirage("ada_mirror",       0.01,  0.20, iters)),
        "MIRAGE-IS(λ=0.01)":   ("mirage", _mirage("mirror_descent",   0.01,  0.05, iters,
                                                   ItakuraSaitoDivergence())),
    }


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def build_dataset_registry() -> dict:
    """Returns {name: (X, y, true_weights_or_None)}."""
    # Alpha signal combination
    X_a, y_a, w_a = toy_alpha_signal_combination(n_samples=300, n_signals=10, seed=42)

    # Sparse portfolio: build regression problem from returns matrix
    ret_p = toy_sparse_portfolio(T=200, N=20, seed=1)
    X_p, y_p = ret_p, ret_p.mean(axis=1)

    # Volatility forecasting
    X_v, y_v, w_v = toy_volatility_forecasting(T=300, n_features=8, seed=7)

    # Ensemble forecasting
    X_e, y_e, w_e = toy_ensemble_forecasting(n_models=15, n_obs=300, seed=21)

    # Macro predictive
    X_m, y_m, w_m = toy_macro_predictive(n_obs=300, n_features=12, seed=31)

    # High-dimensional alpha (stress test: n=50, m=500)
    X_hd, y_hd, w_hd = toy_alpha_signal_combination(n_samples=500, n_signals=50, seed=99)

    return {
        "Alpha(10 signals)":      (X_a,  y_a,  w_a),
        "Portfolio(20 assets)":   (X_p,  y_p,  None),
        "Volatility(8 features)": (X_v,  y_v,  w_v),
        "Ensemble(15 models)":    (X_e,  y_e,  w_e),
        "Macro(12 features)":     (X_m,  y_m,  w_m),
        "HighDim(50 signals)":    (X_hd, y_hd, w_hd),
    }


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_single(model_type, model_obj, X, y, true_weights, n_splits, seed):
    """Run one model with k-fold CV; return dict of aggregated metrics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_mse_list, te_mse_list, r2_list = [], [], []
    ent_list, hhi_list, enb_list, time_list = [], [], [], []
    cos_list, l1_list = [], []

    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        t0 = time.perf_counter()
        if model_type == "sklearn":
            m = copy.deepcopy(model_obj)
            m.fit(Xtr, ytr)
            theta = _to_simplex(m.coef_)
            yhat_tr = Xtr @ theta
            yhat_te = Xte @ theta
        else:
            m = copy.deepcopy(model_obj)
            m.fit(Xtr, ytr)
            theta = m.weights
            yhat_tr = m.predict(Xtr)
            yhat_te = m.predict(Xte)
        elapsed = time.perf_counter() - t0

        tr_mse_list.append(mean_squared_error(ytr, yhat_tr))
        te_mse_list.append(mean_squared_error(yte, yhat_te))
        r2_list.append(r2_score(yte, yhat_te))
        time_list.append(elapsed)
        ent_list.append(entropy(theta))
        hhi_list.append(herfindahl_index(theta))
        enb_list.append(effective_number_of_bets(theta))
        if true_weights is not None:
            cos_list.append(_weight_cosine(theta, true_weights))
            l1_list.append(_weight_l1(theta, true_weights))

    out = {
        "train_mse":      float(np.mean(tr_mse_list)),
        "test_mse":       float(np.mean(te_mse_list)),
        "test_mse_std":   float(np.std(te_mse_list)),
        "test_r2":        float(np.mean(r2_list)),
        "weight_entropy": float(np.mean(ent_list)),
        "hhi":            float(np.mean(hhi_list)),
        "enb":            float(np.mean(enb_list)),
        "time_s":         float(np.mean(time_list)),
    }
    if true_weights is not None:
        out["weight_cosine"] = float(np.mean(cos_list))
        out["weight_l1"]     = float(np.mean(l1_list))
    return out


def run_benchmark(datasets=None, models=None, n_splits=5, seed=42, quick=False):
    """
    Run the full benchmark.

    Returns
    -------
    dict: {dataset_name: {model_name: metrics_dict}}
    """
    all_datasets = build_dataset_registry()
    all_models   = build_model_registry(quick=quick)

    if datasets:
        all_datasets = {k: v for k, v in all_datasets.items()
                        if any(d.lower() in k.lower() for d in datasets)}
    if models:
        all_models = {k: v for k, v in all_models.items()
                      if any(m.lower() in k.lower() for m in models)}

    results = {}
    n_total = len(all_datasets) * len(all_models)
    done = 0

    for ds_name, (X, y, true_w) in all_datasets.items():
        results[ds_name] = {}
        for mdl_name, (mtype, mobj) in all_models.items():
            done += 1
            print(f"  [{done:3d}/{n_total}] {ds_name:30s} | {mdl_name}", end="\r", flush=True)
            try:
                results[ds_name][mdl_name] = run_single(
                    mtype, mobj, X, y, true_w, n_splits, seed
                )
            except Exception as exc:
                results[ds_name][mdl_name] = {"error": str(exc)}

    print()  # clear the \r line
    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".5f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "   ---   "
    return f"{val:{fmt}}"


def print_table(results: dict, metric: str = "test_mse", title: str = None):
    """Print a comparison table for one metric across all datasets × models."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

    all_models = []
    for ds_res in results.values():
        for m in ds_res:
            if m not in all_models:
                all_models.append(m)

    ds_names = list(results.keys())
    col_w = 22
    ds_w  = 30

    # Header
    header = f"{'Model':<{col_w}}" + "".join(f"{d[:ds_w-2]:>{ds_w}}" for d in ds_names)
    print(header)
    print("-" * len(header))

    # Rows — find best per column to bold
    best_per_ds = {}
    for ds in ds_names:
        vals = []
        for m in all_models:
            v = results[ds].get(m, {}).get(metric, None)
            if v is not None and not isinstance(v, str):
                vals.append(v)
        if vals:
            best_per_ds[ds] = min(vals) if "mse" in metric or "l1" in metric or "hhi" in metric \
                               else max(vals)

    # Separator between sklearn and MIRAGE
    mirage_started = False
    for m in all_models:
        is_mirage = m.startswith("MIRAGE")
        if is_mirage and not mirage_started:
            print("-" * len(header))
            mirage_started = True
        row = f"{m:<{col_w}}"
        for ds in ds_names:
            val = results[ds].get(m, {}).get(metric, None)
            if isinstance(val, str):          # error
                cell = f"{'ERR':>{ds_w}}"
            elif val is None:
                cell = f"{'---':>{ds_w}}"
            else:
                marker = "*" if abs(val - best_per_ds.get(ds, float("nan"))) < 1e-9 else " "
                cell = f"{marker}{val:>{ds_w - 1}.5f}"
            row += cell
        print(row)
    print()


def print_full_report(results: dict):
    print("\n" + "=" * 80)
    print("  MIRAGE++ BENCHMARK REPORT")
    print("=" * 80)
    print("  * = best in column\n")

    print_table(results, "test_mse",       "OUT-OF-SAMPLE MSE  (lower is better)")
    print_table(results, "test_r2",        "OUT-OF-SAMPLE R²   (higher is better)")
    print_table(results, "weight_entropy", "WEIGHT ENTROPY H(θ)  (higher = more diversified)")
    print_table(results, "hhi",            "HERFINDAHL INDEX  (lower = more diversified)")
    print_table(results, "enb",            "EFFECTIVE NUMBER OF BETS  (higher = more diversified)")
    print_table(results, "time_s",         "TRAINING TIME (seconds, lower is better)")

    # Weight recovery only for datasets with ground truth
    has_recovery = any(
        "weight_cosine" in list(ds_res.values())[0]
        for ds_res in results.values()
        if ds_res
    )
    if has_recovery:
        print_table(results, "weight_cosine",
                    "WEIGHT RECOVERY: cosine similarity to true θ  (higher is better)")
        print_table(results, "weight_l1",
                    "WEIGHT RECOVERY: L1 distance from true θ  (lower is better)")


def print_summary_ranking(results: dict):
    """Print win-count ranking across all datasets on test MSE."""
    print("\n" + "=" * 60)
    print("  RANKING by out-of-sample MSE (wins across datasets)")
    print("=" * 60)
    wins = {}
    for ds_res in results.values():
        valid = {m: v["test_mse"] for m, v in ds_res.items()
                 if "test_mse" in v and not isinstance(v.get("test_mse"), str)}
        if not valid:
            continue
        best_mse = min(valid.values())
        for m, v in valid.items():
            if abs(v - best_mse) < 1e-9:
                wins[m] = wins.get(m, 0) + 1
    n_datasets = len(results)
    for m, w in sorted(wins.items(), key=lambda x: -x[1]):
        bar = "█" * w + "░" * (n_datasets - w)
        print(f"  {m:<30s}  {bar}  {w}/{n_datasets}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIRAGE++ benchmark")
    parser.add_argument("--dataset", nargs="*", help="Filter datasets by substring")
    parser.add_argument("--model",   nargs="*", help="Filter models by substring")
    parser.add_argument("--folds",   type=int, default=5, help="CV folds (default 5)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--quick",   action="store_true",
                        help="Fewer iterations for faster run")
    args = parser.parse_args()

    print(f"\nMIRAGE++ Benchmark  |  folds={args.folds}  quick={args.quick}")
    print("-" * 60)

    results = run_benchmark(
        datasets=args.dataset,
        models=args.model,
        n_splits=args.folds,
        seed=args.seed,
        quick=args.quick,
    )

    print_full_report(results)
    print_summary_ranking(results)
