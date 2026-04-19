# API Reference

## Core Model

### `MirrorLinearRegression`

```python
from mirror_linear_regression import MirrorLinearRegression
```

Entropy-regularised linear regression on the probability simplex, optimised via mirror descent and its variants.

#### Constructor

```python
MirrorLinearRegression(
    learning_rate: float = 0.1,
    n_iters: int = 500,
    lam: float = 0.05,
    tol: float = 1e-7,
    optimizer: str = "mirror_descent",
    divergence: BregmanDivergence | None = None,
    verbose: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 0.1 | Step size $\eta$ for the mirror descent update |
| `n_iters` | int | 500 | Maximum optimisation iterations |
| `lam` | float | 0.05 | Entropy regularisation $\lambda$; higher = more diversified |
| `tol` | float | 1e-7 | Early-stopping tolerance on absolute loss improvement |
| `optimizer` | str | `"mirror_descent"` | Solver: `"mirror_descent"` \| `"natural_gradient"` \| `"mirror_prox"` \| `"ada_mirror"` |
| `divergence` | BregmanDivergence | None | Bregman geometry; `None` defaults to `KLDivergence()` |
| `verbose` | bool | False | Print loss at each iteration |

#### Methods

**`fit(X, y) -> self`**  
Fit the model. `X`: $(m \times n)$ float array. `y`: $(m,)$ float array.

**`predict(X) -> ndarray`**  
Compute $\hat{y} = X\theta$. Returns $(m,)$ array.

**`convergence_summary() -> dict`**  
Returns `{"n_iters_run", "final_loss", "converged", "final_entropy", "final_enb"}`.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `weights` | ndarray $(n,)$ | Learned simplex weights $\theta$ (read-only) |
| `loss_history` | list[float] | Loss at each iteration |
| `entropy_history` | list[float] | $H(\theta_t)$ at each iteration |
| `mse_history` | list[float] | MSE component at each iteration |

#### Example

```python
from mirror_linear_regression import MirrorLinearRegression

model = MirrorLinearRegression(
    optimizer="mirror_descent",
    lam=0.05,
    learning_rate=0.1,
    n_iters=500,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.weights)            # simplex weights, sum = 1
print(model.convergence_summary())
```

---

## Applications

### `PortfolioAllocator`

```python
from mirror_linear_regression import PortfolioAllocator
```

Fits simplex allocation weights over $N$ assets by minimising $\mathcal{L}(\theta) = \|R\theta - \bar{r}\|^2 - \lambda H(\theta)$, where $R$ is the $(T \times N)$ return matrix and $\bar{r} = R\mathbf{1}/N$ is the equal-weight benchmark.

```python
PortfolioAllocator(
    learning_rate: float = 0.1,
    n_iters: int = 1000,
    lam: float = 0.05,
    optimizer: str = "mirror_descent",
    divergence: BregmanDivergence | None = None,
)
```

**`fit(returns: ndarray) -> self`**  
`returns`: $(T \times N)$ return matrix.

**`get_weights() -> ndarray`**  
Returns $(N,)$ portfolio weights summing to 1.

**`portfolio_diagnostics() -> dict`**  
Returns `{"herfindahl_index", "effective_bets", "diversification_ratio", "weight_entropy", "convergence"}`.

---

### `AlphaSignalCombiner`

```python
from mirror_linear_regression import AlphaSignalCombiner
```

Fits simplex combination weights over $N$ alpha signals. Prevents over-concentration on a single signal — a common failure mode when signals are correlated.

```python
AlphaSignalCombiner(
    learning_rate: float = 0.1,
    n_iters: int = 1000,
    lam: float = 0.05,
    optimizer: str = "mirror_descent",
    divergence: BregmanDivergence | None = None,
)
```

**`fit(X, y) -> self`**  
`X`: $(T \times N)$ signal matrix. `y`: $(T,)$ realised returns.

**`predict(X) -> ndarray`**  
Returns $(T,)$ combined signal predictions.

**`get_signal_weights() -> ndarray`**  
Returns $(N,)$ signal weights summing to 1.

**`signal_diagnostics() -> dict`**  
Returns `{"herfindahl_index", "effective_signals", "diversification_ratio", "weight_entropy", "convergence"}`.

---

## Backtesting

### `RollingPortfolioBacktest`

```python
from mirror_linear_regression import RollingPortfolioBacktest
```

Re-fits `PortfolioAllocator` on a rolling window of return observations, tracking weights, diversification, turnover, and realised P&L over time.

```python
RollingPortfolioBacktest(
    window: int = 252,
    refit_freq: int = 21,
    lam: float = 0.05,
    optimizer: str = "mirror_descent",
    n_iters: int = 400,
    learning_rate: float = 0.1,
)
```

**`run(returns, dates=None) -> BacktestResult`**  
`returns`: $(T \times N)$ array. `dates`: optional $(T,)$ date labels.

### `RollingSignalBacktest`

```python
from mirror_linear_regression import RollingSignalBacktest
```

Rolling re-fit of `AlphaSignalCombiner`, with per-window OOS MSE tracking.

```python
RollingSignalBacktest(
    window: int = 252,
    refit_freq: int = 21,
    lam: float = 0.05,
    optimizer: str = "mirror_descent",
    n_iters: int = 400,
    learning_rate: float = 0.1,
)
```

**`run(X, y, dates=None) -> BacktestResult`**

### `BacktestResult`

Dataclass returned by both backtest runners.

| Field | Shape | Description |
|-------|-------|-------------|
| `weights` | $(P \times N)$ | Per-period portfolio / signal weights |
| `dates` | $(P,)$ | Period end-dates |
| `realised_returns` | $(P,)$ | Realised return per rebalancing period |
| `enb` | $(P,)$ | Effective number of bets per period |
| `hhi` | $(P,)$ | Herfindahl index per period |
| `turnover` | $(P,)$ | L1 weight change vs previous period |
| `mse_per_period` | $(P,)$ | OOS MSE (signal backtest only) |

**`summary() -> dict`**  
Scalar statistics: `n_periods`, `mean_enb`, `mean_hhi`, `mean_turnover`, `total_return`, `sharpe`, `max_drawdown`, `mean_oos_mse`.

**`print_summary()`**  
Prints formatted summary table.

---

## Bregman Divergences

```python
from mirror_linear_regression.bregman import (
    KLDivergence,
    SquaredEuclideanDivergence,
    ItakuraSaitoDivergence,
    BetaDivergence,
    get_divergence,
)
```

All divergences inherit from `BregmanDivergence` and implement:

| Method | Signature | Description |
|--------|-----------|-------------|
| `generator(p)` | `ndarray -> float` | Generator $\phi(p)$ |
| `grad_generator(p)` | `ndarray -> ndarray` | Mirror map $\nabla\phi(p)$ |
| `inverse_mirror(z)` | `ndarray -> ndarray` | Inverse mirror map $(\nabla\phi)^{-1}(z)$ |
| `project(p)` | `ndarray -> ndarray` | Project onto feasible set |
| `divergence(p, q)` | `ndarray, ndarray -> float` | $D_\phi(p \| q)$ |
| `mirror_step(grad, theta, eta)` | — | One mirror descent step |

**Factory function:**

```python
div = get_divergence("kl")           # KLDivergence()
div = get_divergence("euclidean")    # SquaredEuclideanDivergence()
div = get_divergence("itakura_saito")
div = get_divergence("beta", beta=1.5)
```

---

## Geometry

```python
from mirror_linear_regression.geometry import (
    natural_gradient,
    natural_gradient_step,
    fisher_rao_distance,
    geodesic,
    exponential_map,
    logarithmic_map,
    spd_riemannian_gradient,
    spd_retraction,
    spd_geodesic,
    affine_invariant_distance,
    project_to_spd,
)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `natural_gradient(g, theta)` | `ndarray, ndarray -> ndarray` | Fisher-Rao natural gradient |
| `fisher_rao_distance(p, q)` | `ndarray, ndarray -> float` | $2\arccos(\sum\sqrt{p_i q_i})$ |
| `geodesic(p, q, t)` | `ndarray, ndarray, float -> ndarray` | Simplex geodesic at parameter $t$ |
| `spd_riemannian_gradient(eg, sigma)` | `ndarray, ndarray -> ndarray` | $\Sigma \cdot \text{sym}(\nabla f) \cdot \Sigma$ |
| `spd_retraction(sigma, rgrad, eta)` | — | Geodesic retraction on $\text{Sym}^+$ |
| `affine_invariant_distance(A, B)` | — | $\|\log(A^{-1/2}B A^{-1/2})\|_F$ |

---

## Convergence Utilities

```python
from mirror_linear_regression.convergence import (
    kl_regret_bound,
    euclidean_regret_bound,
    optimal_learning_rate,
    strong_convexity_trajectory,
    compute_regret,
    convergence_rate_estimate,
    print_convergence_report,
)
```

| Function | Returns | Description |
|----------|---------|-------------|
| `kl_regret_bound(T, n, G, eta=None)` | float | $G\sqrt{2T\log n}$ at optimal $\eta$ |
| `euclidean_regret_bound(T, n, G)` | float | $G\sqrt{2nT}$ |
| `optimal_learning_rate(T, n, G)` | float | $\sqrt{2\log n/(TG^2)}$ |
| `strong_convexity_trajectory(gap0, mu, eta, num_steps)` | ndarray | $\text{gap}_0 \cdot e^{-\mu\eta t}$ |
| `compute_regret(losses, opt_loss)` | ndarray | Cumulative regret $\sum_t (\mathcal{L}_t - \mathcal{L}^*)$ |
| `convergence_rate_estimate(losses)` | float | Power-law exponent from log-linear fit |

---

## Utility Functions

```python
from mirror_linear_regression.utils_math import (
    entropy,
    kl_divergence,
    herfindahl_index,
    effective_number_of_bets,
    diversification_ratio,
    hellinger_distance,
    bhattacharyya_coefficient,
    project_simplex,
    normalize,
    softmax,
)
```

| Function | Formula | Description |
|----------|---------|-------------|
| `entropy(p)` | $-\sum p_i \log p_i$ | Shannon entropy (nats) |
| `herfindahl_index(p)` | $\sum p_i^2$ | Concentration index $\in [1/n, 1]$ |
| `effective_number_of_bets(p)` | $\exp(H(p))$ | Equivalent uniform portfolio size |
| `diversification_ratio(p)` | $H(p) / \log n$ | Normalised entropy $\in [0,1]$ |
| `hellinger_distance(p, q)` | $\frac{1}{\sqrt{2}}\|\sqrt{p}-\sqrt{q}\|$ | Symmetric divergence |
| `project_simplex(v)` | — | Wang & Carreira-Perpinan (2013) |

---

## Data Loaders

```python
from examples.market_datasets import (
    load_sector_etf_portfolio,
    load_fama_french_signals,
    load_technical_signals,
    load_equity_universe,
    describe,
)
```

All loaders return a `dict` with keys: `X` (ndarray), `y` (ndarray), `n_samples`, `n_features`, `description`, and dataset-specific keys (e.g., `tickers`, `factor_names`, `signal_names`, `dates`).

Data is cached to `data/` on first call; subsequent calls load from parquet without network access.

```python
from examples.synthetic_datasets import (
    toy_alpha_signal_combination,
    toy_sparse_portfolio,
    toy_volatility_forecasting,
    toy_factor_exposures,
    toy_ensemble_forecasting,
    toy_option_surface,
    toy_macro_predictive,
    large_alpha_signals,
    large_portfolio,
    correlated_signals,
    heavy_tail_returns,
    nonstationary_signals,
    sparse_true_weights,
    adversarial_collinear,
)
```
