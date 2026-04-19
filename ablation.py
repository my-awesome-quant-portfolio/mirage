"""
ablation.py — Systematic ablation studies for MIRAGE++.

Each study isolates one design dimension while keeping all others fixed,
following standard NeurIPS ablation methodology.

Studies
-------
1. lambda_sweep       Effect of entropy regularisation strength on MSE / entropy / HHI
2. optimizer_compare  Mirror descent vs Natural gradient vs Mirror-Prox vs AdaMirror
3. divergence_compare KL vs Euclidean vs Itakura-Saito vs Beta(1.5) vs Beta(2.0)
4. lr_sensitivity     Learning rate grid: sensitivity and stability analysis
5. dim_scaling        Dimension n from 5 → 500: regret bound vs empirical MSE
6. sample_scaling     Sample size m from 50 → 10 000: generalisation gap
7. noise_robustness   SNR from 0.1 → 100: MSE and weight recovery vs noise level
8. corr_robustness    Feature correlation rho from 0 → 0.95: stability under multicollinearity

Run:
    python ablation.py                      # all studies
    python ablation.py --study lambda       # single study
    python ablation.py --study dim --quick  # fast version
"""

import sys
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from mirror_linear_regression import MirrorLinearRegression
from mirror_linear_regression.bregman import (
    KLDivergence, SquaredEuclideanDivergence,
    ItakuraSaitoDivergence, BetaDivergence,
)
from mirror_linear_regression.convergence import (
    kl_regret_bound, euclidean_regret_bound,
    convergence_rate_estimate,
)
from mirror_linear_regression.utils_math import (
    entropy, herfindahl_index, effective_number_of_bets,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_simplex_data(n, m, snr=20.0, seed=0, rho=0.0):
    """
    Generate (X, y, true_theta) where X has feature correlation rho.
    true_theta is a Dirichlet sample on the n-simplex.
    SNR = var(signal) / var(noise).
    """
    rng = np.random.RandomState(seed)
    # Correlated feature matrix via Cholesky
    cov = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    L = np.linalg.cholesky(cov)
    Z = rng.randn(m, n)
    X = Z @ L.T
    true_theta = rng.dirichlet(np.ones(n))
    signal = X @ true_theta
    noise_var = np.var(signal) / snr
    y = signal + rng.randn(m) * np.sqrt(noise_var)
    return X, y, true_theta


def _weight_cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _mse(model, X, y):
    return float(np.mean((model.predict(X) - y) ** 2))


# ---------------------------------------------------------------------------
# Study 1: Lambda sweep
# ---------------------------------------------------------------------------

def ablation_lambda(quick=False):
    """
    Vary entropy regularisation strength lambda from 1e-4 to 5.0.
    Fixed: n=10, m=200, mirror_descent, lr=0.1, 500 iters.
    """
    lambdas = np.logspace(-4, 0.7, 20 if not quick else 8)
    n, m = 10, 200
    X, y, true_w = _make_simplex_data(n, m, snr=20.0, seed=42)
    X_test, y_test, _ = _make_simplex_data(n, 200, snr=20.0, seed=99)

    rows = []
    for lam in lambdas:
        model = MirrorLinearRegression(
            lam=lam, learning_rate=0.1, n_iters=500, tol=1e-8
        ).fit(X, y)
        w = model.weights
        rows.append({
            "lambda":         lam,
            "train_mse":      _mse(model, X, y),
            "test_mse":       _mse(model, X_test, y_test),
            "weight_entropy": entropy(w),
            "hhi":            herfindahl_index(w),
            "enb":            effective_number_of_bets(w),
            "weight_cosine":  _weight_cosine(w, true_w),
            "n_iters":        model.n_iters_run,
        })
    return rows


# ---------------------------------------------------------------------------
# Study 2: Optimizer comparison
# ---------------------------------------------------------------------------

def ablation_optimizer(quick=False):
    """
    Compare all four MIRAGE++ optimisers on identical data with identical
    learning rate and iteration budget.
    """
    n, m = 15, 300
    X, y, true_w = _make_simplex_data(n, m, snr=30.0, seed=7)
    X_te, y_te, _ = _make_simplex_data(n, 200, snr=30.0, seed=77)

    configs = {
        "mirror_descent":   dict(optimizer="mirror_descent",   learning_rate=0.1),
        "natural_gradient": dict(optimizer="natural_gradient", learning_rate=0.05),
        "mirror_prox":      dict(optimizer="mirror_prox",      learning_rate=0.1),
        "ada_mirror":       dict(optimizer="ada_mirror",       learning_rate=0.2),
    }

    iters = 300 if quick else 700
    rows = []
    for name, cfg in configs.items():
        model = MirrorLinearRegression(
            lam=0.01, n_iters=iters, tol=1e-9, **cfg
        ).fit(X, y)
        w = model.weights
        rows.append({
            "optimizer":      name,
            "train_mse":      _mse(model, X, y),
            "test_mse":       _mse(model, X_te, y_te),
            "weight_entropy": entropy(w),
            "hhi":            herfindahl_index(w),
            "weight_cosine":  _weight_cosine(w, true_w),
            "n_iters":        model.n_iters_run,
            "final_loss":     model.loss_history[-1],
            "conv_rate":      convergence_rate_estimate(model.loss_history),
        })
    return rows


# ---------------------------------------------------------------------------
# Study 3: Divergence comparison
# ---------------------------------------------------------------------------

def ablation_divergence(quick=False):
    """
    Compare four Bregman divergences with mirror descent.
    All other hyperparameters fixed.
    """
    n, m = 12, 250
    X, y, true_w = _make_simplex_data(n, m, snr=25.0, seed=3)
    X_te, y_te, _ = _make_simplex_data(n, 200, snr=25.0, seed=33)

    divergences = {
        "KL":             (KLDivergence(),              0.10),
        "Euclidean":      (SquaredEuclideanDivergence(), 0.05),
        "Itakura-Saito":  (ItakuraSaitoDivergence(),    0.05),
        "Beta(β=1.5)":    (BetaDivergence(beta=1.5),    0.05),
        "Beta(β=2.0)":    (BetaDivergence(beta=2.0),    0.02),
    }

    iters = 300 if quick else 600
    rows = []
    for name, (div, lr) in divergences.items():
        model = MirrorLinearRegression(
            optimizer="mirror_descent", divergence=div,
            lam=0.01, learning_rate=lr, n_iters=iters, tol=1e-9,
        ).fit(X, y)
        w = model.weights
        rows.append({
            "divergence":     name,
            "train_mse":      _mse(model, X, y),
            "test_mse":       _mse(model, X_te, y_te),
            "weight_entropy": entropy(w),
            "hhi":            herfindahl_index(w),
            "weight_cosine":  _weight_cosine(w, true_w),
            "final_loss":     model.loss_history[-1],
            "conv_rate":      convergence_rate_estimate(model.loss_history),
        })
    return rows


# ---------------------------------------------------------------------------
# Study 4: Learning rate sensitivity
# ---------------------------------------------------------------------------

def ablation_learning_rate(quick=False):
    """
    Sweep learning rate from 1e-3 to 2.0 (log-spaced).
    Report: final test MSE, convergence rate, and whether training diverged.
    """
    lrs = np.logspace(-3, 0.3, 15 if not quick else 6)
    n, m = 10, 200
    X, y, true_w = _make_simplex_data(n, m, snr=20.0, seed=11)
    X_te, y_te, _ = _make_simplex_data(n, 200, snr=20.0, seed=111)

    rows = []
    for lr in lrs:
        model = MirrorLinearRegression(
            lam=0.01, learning_rate=lr, n_iters=500, tol=1e-9
        ).fit(X, y)
        w = model.weights
        loss_ok = all(np.isfinite(l) for l in model.loss_history)
        rows.append({
            "lr":             lr,
            "train_mse":      _mse(model, X, y),
            "test_mse":       _mse(model, X_te, y_te),
            "weight_entropy": entropy(w),
            "conv_rate":      convergence_rate_estimate(model.loss_history),
            "n_iters":        model.n_iters_run,
            "stable":         loss_ok and np.isfinite(_mse(model, X, y)),
        })
    return rows


# ---------------------------------------------------------------------------
# Study 5: Dimensionality scaling
# ---------------------------------------------------------------------------

def ablation_dim_scaling(quick=False):
    """
    Vary n (simplex dimension) from 5 to 500 with m = 5*n (fixed ratio).
    Report: test MSE, empirical regret rate, KL bound vs Euclidean bound.
    """
    dims = [5, 10, 20, 50, 100, 200, 500] if not quick else [5, 10, 20, 50]
    iters = 400

    rows = []
    for n in dims:
        m = 5 * n
        X, y, true_w = _make_simplex_data(n, m, snr=20.0, seed=n)
        X_te, y_te, _ = _make_simplex_data(n, m, snr=20.0, seed=n + 1000)

        model = MirrorLinearRegression(
            lam=0.01, learning_rate=0.1, n_iters=iters, tol=1e-9
        ).fit(X, y)
        w = model.weights
        T = model.n_iters_run

        rows.append({
            "n":              n,
            "m":              m,
            "test_mse":       _mse(model, X_te, y_te),
            "weight_entropy": entropy(w),
            "hhi":            herfindahl_index(w),
            "weight_cosine":  _weight_cosine(w, true_w),
            "n_iters":        T,
            "kl_bound":       kl_regret_bound(T, n),
            "eucl_bound":     euclidean_regret_bound(T, n),
            "kl_vs_eucl":     kl_regret_bound(T, n) / max(euclidean_regret_bound(T, n), 1e-9),
        })
    return rows


# ---------------------------------------------------------------------------
# Study 6: Sample size scaling
# ---------------------------------------------------------------------------

def ablation_sample_scaling(quick=False):
    """
    Fix n=20, vary m from 50 to 10 000.
    Report: train MSE, test MSE, generalisation gap = test_mse - train_mse.
    """
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000] \
        if not quick else [50, 100, 200, 500, 1000]
    n = 20

    rows = []
    for m in sample_sizes:
        X, y, true_w = _make_simplex_data(n, m, snr=20.0, seed=m)
        m_test = min(m, 500)
        X_te, y_te, _ = _make_simplex_data(n, m_test, snr=20.0, seed=m + 9999)

        model = MirrorLinearRegression(
            lam=0.01, learning_rate=0.1, n_iters=500, tol=1e-9
        ).fit(X, y)
        tr_mse = _mse(model, X, y)
        te_mse = _mse(model, X_te, y_te)

        rows.append({
            "m":             m,
            "train_mse":     tr_mse,
            "test_mse":      te_mse,
            "gen_gap":       te_mse - tr_mse,
            "weight_cosine": _weight_cosine(model.weights, true_w),
            "n_iters":       model.n_iters_run,
        })
    return rows


# ---------------------------------------------------------------------------
# Study 7: Noise robustness (SNR sweep)
# ---------------------------------------------------------------------------

def ablation_noise(quick=False):
    """
    Fix n=10, m=200.  Vary SNR from 0.5 to 500 (log-spaced).
    Compare MIRAGE++ (λ=0.05) vs OLS-simplex on weight recovery and MSE.
    """
    from sklearn.linear_model import LinearRegression

    def _ols_simplex(X, y):
        coef = LinearRegression(fit_intercept=False).fit(X, y).coef_
        coef = np.clip(coef, 0, None)
        s = coef.sum()
        return coef / s if s > 1e-12 else np.ones(len(coef)) / len(coef)

    snrs = np.logspace(-0.3, 2.7, 16 if not quick else 6)
    n, m = 10, 200

    rows = []
    for snr in snrs:
        X, y, true_w = _make_simplex_data(n, m, snr=snr, seed=5)
        X_te, y_te, _ = _make_simplex_data(n, 200, snr=snr, seed=55)

        # MIRAGE++
        model = MirrorLinearRegression(
            lam=0.05, learning_rate=0.1, n_iters=500, tol=1e-9
        ).fit(X, y)
        w_m = model.weights

        # OLS (projected to simplex)
        w_ols = _ols_simplex(X, y)

        rows.append({
            "snr":                  snr,
            "mirage_test_mse":      _mse(model, X_te, y_te),
            "ols_test_mse":         float(np.mean((X_te @ w_ols - y_te) ** 2)),
            "mirage_weight_cosine": _weight_cosine(w_m, true_w),
            "ols_weight_cosine":    _weight_cosine(w_ols, true_w),
            "mirage_entropy":       entropy(w_m),
            "ols_entropy":          entropy(w_ols),
        })
    return rows


# ---------------------------------------------------------------------------
# Study 8: Correlation robustness
# ---------------------------------------------------------------------------

def ablation_correlation(quick=False):
    """
    Fix n=10, m=300.  Vary feature correlation rho from 0 to 0.95.
    Report: test MSE, weight cosine similarity, HHI.
    """
    rhos = np.linspace(0.0, 0.95, 12 if not quick else 5)
    n, m = 10, 300

    rows = []
    for rho in rhos:
        X, y, true_w = _make_simplex_data(n, m, snr=20.0, rho=rho, seed=13)
        X_te, y_te, _ = _make_simplex_data(n, 200, snr=20.0, rho=rho, seed=130)

        model = MirrorLinearRegression(
            lam=0.01, learning_rate=0.1, n_iters=500, tol=1e-9
        ).fit(X, y)
        w = model.weights

        rows.append({
            "rho":            rho,
            "test_mse":       _mse(model, X_te, y_te),
            "weight_entropy": entropy(w),
            "hhi":            herfindahl_index(w),
            "weight_cosine":  _weight_cosine(w, true_w),
            "n_iters":        model.n_iters_run,
        })
    return rows


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _hdr(title, width=70):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def _row(cells, widths):
    return "  ".join(f"{str(c):<{w}}" if i == 0 else f"{str(c):>{w}}"
                     for i, (c, w) in enumerate(zip(cells, widths)))


def print_lambda(rows):
    _hdr("ABLATION 1: Lambda (entropy regularisation strength)")
    print(_row(["lambda", "train_mse", "test_mse", "H(θ)", "HHI", "ENB", "cos(θ,θ*)"],
               [10, 10, 10, 8, 8, 6, 10]))
    print("-" * 70)
    for r in rows:
        print(_row([
            f"{r['lambda']:.5f}", f"{r['train_mse']:.5f}", f"{r['test_mse']:.5f}",
            f"{r['weight_entropy']:.4f}", f"{r['hhi']:.4f}",
            f"{r['enb']:.2f}", f"{r['weight_cosine']:.4f}",
        ], [10, 10, 10, 8, 8, 6, 10]))


def print_optimizer(rows):
    _hdr("ABLATION 2: Optimizer comparison")
    print(_row(["optimizer", "train_mse", "test_mse", "H(θ)", "cos(θ,θ*)", "iters", "conv_α"],
               [20, 10, 10, 8, 10, 6, 8]))
    print("-" * 80)
    for r in rows:
        print(_row([
            r["optimizer"], f"{r['train_mse']:.5f}", f"{r['test_mse']:.5f}",
            f"{r['weight_entropy']:.4f}", f"{r['weight_cosine']:.4f}",
            r["n_iters"], f"{r['conv_rate']:.3f}",
        ], [20, 10, 10, 8, 10, 6, 8]))


def print_divergence(rows):
    _hdr("ABLATION 3: Bregman divergence comparison")
    print(_row(["divergence", "train_mse", "test_mse", "H(θ)", "cos(θ,θ*)", "conv_α"],
               [18, 10, 10, 8, 10, 8]))
    print("-" * 70)
    for r in rows:
        print(_row([
            r["divergence"], f"{r['train_mse']:.5f}", f"{r['test_mse']:.5f}",
            f"{r['weight_entropy']:.4f}", f"{r['weight_cosine']:.4f}",
            f"{r['conv_rate']:.3f}",
        ], [18, 10, 10, 8, 10, 8]))


def print_lr(rows):
    _hdr("ABLATION 4: Learning rate sensitivity")
    print(_row(["lr", "train_mse", "test_mse", "H(θ)", "conv_α", "iters", "stable"],
               [8, 10, 10, 8, 8, 6, 7]))
    print("-" * 65)
    for r in rows:
        print(_row([
            f"{r['lr']:.5f}", f"{r['train_mse']:.5f}", f"{r['test_mse']:.5f}",
            f"{r['weight_entropy']:.4f}", f"{r['conv_rate']:.3f}",
            r["n_iters"], str(r["stable"]),
        ], [8, 10, 10, 8, 8, 6, 7]))


def print_dim(rows):
    _hdr("ABLATION 5: Dimensionality scaling  (m = 5n)")
    print(_row(["n", "test_mse", "H(θ)", "cos(θ,θ*)", "KL_bound", "Eucl_bound", "KL/Eucl"],
               [6, 10, 8, 10, 10, 10, 8]))
    print("-" * 70)
    for r in rows:
        print(_row([
            r["n"], f"{r['test_mse']:.5f}", f"{r['weight_entropy']:.4f}",
            f"{r['weight_cosine']:.4f}", f"{r['kl_bound']:.3f}",
            f"{r['eucl_bound']:.3f}", f"{r['kl_vs_eucl']:.4f}",
        ], [6, 10, 8, 10, 10, 10, 8]))


def print_sample(rows):
    _hdr("ABLATION 6: Sample size scaling  (n=20 fixed)")
    print(_row(["m", "train_mse", "test_mse", "gen_gap", "cos(θ,θ*)", "iters"],
               [7, 10, 10, 10, 10, 6]))
    print("-" * 60)
    for r in rows:
        print(_row([
            r["m"], f"{r['train_mse']:.5f}", f"{r['test_mse']:.5f}",
            f"{r['gen_gap']:.5f}", f"{r['weight_cosine']:.4f}", r["n_iters"],
        ], [7, 10, 10, 10, 10, 6]))


def print_noise(rows):
    _hdr("ABLATION 7: Noise robustness (SNR sweep)")
    print(_row(["SNR", "MIRAGE_MSE", "OLS_MSE", "MIRAGE_cos", "OLS_cos", "MIRAGE_H"],
               [8, 12, 12, 12, 10, 10]))
    print("-" * 70)
    for r in rows:
        print(_row([
            f"{r['snr']:.2f}", f"{r['mirage_test_mse']:.5f}", f"{r['ols_test_mse']:.5f}",
            f"{r['mirage_weight_cosine']:.4f}", f"{r['ols_weight_cosine']:.4f}",
            f"{r['mirage_entropy']:.4f}",
        ], [8, 12, 12, 12, 10, 10]))


def print_corr(rows):
    _hdr("ABLATION 8: Feature correlation robustness")
    print(_row(["rho", "test_mse", "H(θ)", "HHI", "cos(θ,θ*)", "iters"],
               [6, 10, 8, 8, 10, 6]))
    print("-" * 55)
    for r in rows:
        print(_row([
            f"{r['rho']:.3f}", f"{r['test_mse']:.5f}", f"{r['weight_entropy']:.4f}",
            f"{r['hhi']:.4f}", f"{r['weight_cosine']:.4f}", r["n_iters"],
        ], [6, 10, 8, 8, 10, 6]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STUDY_MAP = {
    "lambda":      (ablation_lambda,      print_lambda),
    "optimizer":   (ablation_optimizer,   print_optimizer),
    "divergence":  (ablation_divergence,  print_divergence),
    "lr":          (ablation_learning_rate, print_lr),
    "dim":         (ablation_dim_scaling, print_dim),
    "sample":      (ablation_sample_scaling, print_sample),
    "noise":       (ablation_noise,       print_noise),
    "corr":        (ablation_correlation, print_corr),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIRAGE++ ablation studies")
    parser.add_argument("--study", nargs="*",
                        help=f"Studies to run: {list(STUDY_MAP.keys())}")
    parser.add_argument("--quick", action="store_true",
                        help="Faster run with fewer grid points / iterations")
    args = parser.parse_args()

    studies = args.study if args.study else list(STUDY_MAP.keys())

    print("\nMIRAGE++ Ablation Studies")
    print(f"Running: {studies}  |  quick={args.quick}\n")

    for name in studies:
        if name not in STUDY_MAP:
            print(f"[WARNING] Unknown study '{name}', skipping.")
            continue
        run_fn, print_fn = STUDY_MAP[name]
        print(f"  Running {name}...", end=" ", flush=True)
        rows = run_fn(quick=args.quick)
        print("done.")
        print_fn(rows)

    print("\nAblation complete.")
