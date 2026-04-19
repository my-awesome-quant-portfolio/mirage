"""
Toy dataset generators for MIRAGE++ quantitative finance use cases.

Small-scale generators (original)
----------------------------------
toy_alpha_signal_combination   -- simplex regression with known true weights
toy_sparse_portfolio           -- returns matrix for portfolio allocation
toy_volatility_forecasting     -- vol surface regression
toy_factor_exposures           -- per-stock factor loading estimation
toy_ensemble_forecasting       -- model ensemble combination
toy_option_surface             -- option implied-vol surface
toy_macro_predictive           -- macro factor prediction

Large-scale / stress-test generators (new)
-------------------------------------------
large_alpha_signals            -- high-dimensional alpha (n=500, m=10 000)
large_portfolio                -- wide portfolio (N=200, T=1000)
correlated_signals             -- signals with AR(1) factor structure (rho control)
heavy_tail_returns             -- student-t returns with fat tails (df control)
nonstationary_signals          -- regime-switching mean / covariance
sparse_true_weights            -- ground truth has k << n nonzero weights
adversarial_collinear          -- near-duplicate features (ill-conditioned X)
"""

import numpy as np


# ===========================================================================
# Original small-scale generators
# ===========================================================================

def toy_alpha_signal_combination(n_samples=200, n_signals=10, noise=0.01, seed=42):
    """
    Linear alpha signal combination with known Dirichlet ground truth weights.

    Returns (X, y, true_weights) where X is (n_samples x n_signals) and
    true_weights sums to 1.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_signals)
    true_weights = rng.dirichlet(np.ones(n_signals))
    y = X @ true_weights + noise * rng.randn(n_samples)
    return X, y, true_weights


def toy_sparse_portfolio(T=30, N=100, noise=0.01, seed=1):
    """
    Synthetic asset return matrix (T periods x N assets).
    Returns: returns array (T x N).
    """
    rng = np.random.RandomState(seed)
    returns = rng.randn(T, N) * 0.02 + 0.001
    return returns


def toy_volatility_forecasting(T=200, n_features=6, seed=7):
    """
    Volatility forecasting: features -> realised vol.
    Returns (X, vol, true_coefs).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(T, n_features)
    true_coefs = rng.uniform(0.1, 0.5, size=n_features)
    vol = np.abs(X @ true_coefs + 0.1 * rng.randn(T))
    return X, vol, true_coefs


def toy_factor_exposures(n_stocks=50, n_factors=8, seed=11):
    """
    Per-stock factor loading matrix.
    Returns (F, exposures) where exposures[i] sums to 1.
    """
    rng = np.random.RandomState(seed)
    F = rng.randn(n_stocks, n_factors)
    exposures = rng.dirichlet(np.ones(n_factors), size=n_stocks)
    return F, exposures


def toy_ensemble_forecasting(n_models=15, n_obs=200, seed=21):
    """
    Ensemble model combination with known ground-truth weights.
    Returns (signals, y, true_weights).
    """
    rng = np.random.RandomState(seed)
    signals = rng.randn(n_obs, n_models)
    true_weights = rng.dirichlet(np.ones(n_models))
    y = signals @ true_weights + 0.05 * rng.randn(n_obs)
    return signals, y, true_weights


def toy_option_surface(n_points=100, seed=17):
    """
    Implied-vol surface points: (strikes, maturities, observed, true_surface).
    """
    rng = np.random.RandomState(seed)
    strikes    = np.linspace(80, 120, n_points)
    maturities = np.linspace(0.1, 2.0, n_points)
    surface    = 0.2 + 0.1 * np.exp(-((strikes - 100) ** 2) / 200) + 0.05 * maturities
    observed   = surface + 0.02 * rng.randn(n_points)
    return strikes, maturities, observed, surface


def toy_macro_predictive(n_obs=150, n_features=12, seed=31):
    """
    Macro factor prediction: X -> y, weights may be negative (not simplex GT).
    Returns (X, y, true_coefs).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_obs, n_features)
    true_coefs = rng.uniform(-0.2, 0.2, size=n_features)
    y = X @ true_coefs + 0.1 * rng.randn(n_obs)
    return X, y, true_coefs


# ===========================================================================
# Large-scale / stress-test generators
# ===========================================================================

def large_alpha_signals(n_signals=500, n_samples=10_000, noise=0.005, seed=100):
    """
    High-dimensional alpha combination (n_signals >> n_samples not required).

    Useful for testing:
    - KL regret advantage over Euclidean (log n vs sqrt n)
    - Runtime scaling
    - Convergence at large n

    Returns (X, y, true_weights).
    """
    rng = np.random.RandomState(seed)
    # Sparse ground truth: 10% non-zero weights
    k = max(1, n_signals // 10)
    raw = np.zeros(n_signals)
    idx = rng.choice(n_signals, k, replace=False)
    raw[idx] = rng.exponential(scale=1.0, size=k)
    true_weights = raw / raw.sum()

    X = rng.randn(n_samples, n_signals)
    y = X @ true_weights + noise * rng.randn(n_samples)
    return X, y, true_weights


def large_portfolio(T=1_000, N=200, seed=200):
    """
    Large-scale portfolio: T=1000 periods, N=200 assets.
    Includes realistic features:
    - Factor structure (5 latent factors)
    - Heteroscedastic volatility
    - Cross-sectional correlation

    Returns returns matrix (T x N).
    """
    rng = np.random.RandomState(seed)
    n_factors = 5
    # Factor model: r = B * f + epsilon
    B = rng.randn(N, n_factors) * 0.1       # factor loadings
    f = rng.randn(T, n_factors) * 0.02      # factor returns
    idio = rng.randn(T, N) * 0.01           # idiosyncratic
    returns = f @ B.T + idio + 0.0005       # small positive drift
    return returns


def correlated_signals(n_signals=50, n_obs=500, rho=0.7, seed=300):
    """
    Signals with strong cross-correlation via latent factor structure.

    Generates n_signals correlated via a single common factor with loading rho.
    Correlation matrix: Sigma_ij = rho + (1-rho) * delta_ij.

    Useful for: testing robustness to multicollinearity.

    Returns (X, y, true_weights).
    """
    rng = np.random.RandomState(seed)
    true_weights = rng.dirichlet(np.ones(n_signals))

    # Equicorrelation structure
    cov = rho * np.ones((n_signals, n_signals)) + (1 - rho) * np.eye(n_signals)
    L = np.linalg.cholesky(cov)
    X = rng.randn(n_obs, n_signals) @ L.T

    noise_var = np.var(X @ true_weights) / 20.0   # SNR = 20
    y = X @ true_weights + rng.randn(n_obs) * np.sqrt(noise_var)
    return X, y, true_weights


def heavy_tail_returns(T=1_000, N=50, df=3, seed=400):
    """
    Asset returns drawn from a multivariate Student-t distribution (fat tails).

    df=3 gives extreme kurtosis (infinite 4th moment).
    Useful for: testing robustness to outliers and non-Gaussian noise.

    Returns returns matrix (T x N).
    """
    rng = np.random.RandomState(seed)
    # Student-t via normal / chi-squared
    Z = rng.randn(T, N)
    chi2 = rng.chisquare(df=df, size=T)
    scale = np.sqrt(df / chi2)
    returns = (Z * scale[:, None]) * 0.02 + 0.001
    return returns


def nonstationary_signals(n_signals=20, T=600, n_regimes=3, seed=500):
    """
    Non-stationary signals with regime switches.

    The data-generating process changes every T//n_regimes steps:
    different signal means, variances, and true weights in each regime.

    Useful for: out-of-sample degradation analysis, regime detection.

    Returns (X, y, regime_labels, per_regime_true_weights).
    """
    rng = np.random.RandomState(seed)
    seg = T // n_regimes
    X_parts, y_parts, labels, weights = [], [], [], []

    for r in range(n_regimes):
        tw = rng.dirichlet(np.ones(n_signals))
        weights.append(tw)

        # Each regime has different signal distribution
        mu = rng.uniform(-0.1, 0.1, n_signals)
        std = rng.uniform(0.5, 2.0, n_signals)
        Xr = rng.randn(seg, n_signals) * std + mu

        noise_var = np.var(Xr @ tw) / 15.0
        yr = Xr @ tw + rng.randn(seg) * np.sqrt(noise_var)

        X_parts.append(Xr)
        y_parts.append(yr)
        labels.extend([r] * seg)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y, np.array(labels), weights


def sparse_true_weights(n=100, k=5, n_obs=500, snr=20.0, seed=600):
    """
    True weight vector is k-sparse (only k of n weights non-zero).

    Tests whether entropy regularisation can recover approximately-sparse
    solutions — the answer is no (entropy pushes toward uniform), which
    is an intentional contrast with Lasso.

    Returns (X, y, true_weights) where true_weights has exactly k non-zeros.
    """
    rng = np.random.RandomState(seed)
    raw = np.zeros(n)
    idx = rng.choice(n, k, replace=False)
    raw[idx] = rng.exponential(1.0, k)
    true_weights = raw / raw.sum()

    X = rng.randn(n_obs, n)
    noise_var = np.var(X @ true_weights) / snr
    y = X @ true_weights + rng.randn(n_obs) * np.sqrt(noise_var)
    return X, y, true_weights


def adversarial_collinear(n=30, n_dupes=10, n_obs=300, seed=700):
    """
    Near-duplicate features: ill-conditioned X.

    Creates n_dupes pairs of nearly identical columns (differing by epsilon
    noise), making X^T X nearly singular.  Tests numerical stability of the
    mirror descent update.

    Returns (X, y, true_weights).
    """
    rng = np.random.RandomState(seed)
    n_base = n - n_dupes
    true_weights = rng.dirichlet(np.ones(n))

    X_base = rng.randn(n_obs, n_base)
    # Duplicate the first n_dupes base columns with tiny perturbation
    X_dupes = X_base[:, :n_dupes] + 1e-3 * rng.randn(n_obs, n_dupes)
    X = np.hstack([X_base, X_dupes])

    noise_var = np.var(X @ true_weights) / 10.0
    y = X @ true_weights + rng.randn(n_obs) * np.sqrt(noise_var)
    return X, y, true_weights
