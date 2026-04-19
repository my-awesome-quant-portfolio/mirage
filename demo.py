"""
MIRAGE++ end-to-end demonstration.

Covers:
  1. Core model with all four optimisers
  2. Custom Bregman divergences
  3. Portfolio allocation with diagnostics
  4. Alpha signal combination
  5. Rolling backtest
  6. Convergence analysis utilities
  7. Riemannian geometry operations
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mirror_linear_regression import (
    MirrorLinearRegression,
    PortfolioAllocator,
    AlphaSignalCombiner,
    RollingPortfolioBacktest,
    RollingSignalBacktest,
)
from mirror_linear_regression.bregman import get_divergence
from mirror_linear_regression.convergence import (
    kl_regret_bound,
    euclidean_regret_bound,
    optimal_learning_rate,
    print_convergence_report,
)
from mirror_linear_regression.geometry import (
    fisher_rao_distance,
    geodesic,
    natural_gradient,
)
from mirror_linear_regression.utils_math import (
    entropy,
    effective_number_of_bets,
    herfindahl_index,
)

rng = np.random.RandomState(42)

# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------
m, n = 300, 15
X = rng.randn(m, n)
w_true = rng.dirichlet(np.ones(n))        # ground-truth simplex weights
y = X @ w_true + 0.02 * rng.randn(m)

print("=" * 60)
print("MIRAGE++ — End-to-End Demonstration")
print("=" * 60)
print(f"\nDataset: {m} observations, {n} features")
print(f"Ground-truth entropy H(w*)  = {entropy(w_true):.3f} nats")
print(f"Ground-truth ENB            = {effective_number_of_bets(w_true):.2f}")

# ---------------------------------------------------------------------------
# 1. Core model — all four optimisers
# ---------------------------------------------------------------------------
print("\n--- 1. Core model: all four optimisers ---")

optimiser_cfg = [
    ("mirror_descent",   0.10),
    ("natural_gradient", 0.05),
    ("mirror_prox",      0.10),
    ("ada_mirror",       0.20),
]

for opt, lr in optimiser_cfg:
    model = MirrorLinearRegression(
        optimizer=opt,
        lam=0.05,
        learning_rate=lr,
        n_iters=600,
    )
    model.fit(X, y)
    cs = model.convergence_summary()
    residuals = y - model.predict(X)
    mse = float(np.mean(residuals ** 2))
    cosine = float(np.dot(model.weights, w_true) /
                   (np.linalg.norm(model.weights) * np.linalg.norm(w_true)))
    print(f"  {opt:<20s}  iters={cs['n_iters_run']:3d}  mse={mse:.5f}"
          f"  H={cs['final_entropy']:.2f}  ENB={cs['final_enb']:.1f}"
          f"  cos(w,w*)={cosine:.3f}")

# ---------------------------------------------------------------------------
# 2. Custom Bregman divergences
# ---------------------------------------------------------------------------
print("\n--- 2. Custom Bregman divergences ---")

div_configs = [
    ("KL (default)",    None),
    ("Euclidean",       get_divergence("euclidean")),
    ("Itakura-Saito",   get_divergence("itakura_saito")),
    ("Beta (b=1.5)",    get_divergence("beta", beta=1.5)),
]

for label, div in div_configs:
    model = MirrorLinearRegression(lam=0.02, n_iters=500, divergence=div)
    model.fit(X, y)
    mse = float(np.mean((y - model.predict(X)) ** 2))
    print(f"  {label:<20s}  mse={mse:.5f}  H={entropy(model.weights):.3f}"
          f"  HHI={herfindahl_index(model.weights):.4f}")

# ---------------------------------------------------------------------------
# 3. Lambda sensitivity
# ---------------------------------------------------------------------------
print("\n--- 3. Lambda sensitivity ---")

lambdas = np.logspace(-3, 0, 8)
for lam in lambdas:
    model = MirrorLinearRegression(lam=lam, n_iters=400)
    model.fit(X, y)
    mse = float(np.mean((y - model.predict(X)) ** 2))
    enb = effective_number_of_bets(model.weights)
    print(f"  lambda={lam:.4f}  mse={mse:.5f}  ENB={enb:.2f}  "
          f"max_weight={model.weights.max():.3f}")

# ---------------------------------------------------------------------------
# 4. Portfolio allocation
# ---------------------------------------------------------------------------
print("\n--- 4. Portfolio allocation ---")

T_port, N_port = 500, 12
returns_sim = rng.randn(T_port, N_port) * 0.01 + 0.0003

allocator = PortfolioAllocator(
    lam=0.1,
    optimizer="ada_mirror",
    n_iters=800,
)
allocator.fit(returns_sim)
weights_port = allocator.get_weights()
diag = allocator.portfolio_diagnostics()

print(f"  Weights (first 5): {np.round(weights_port[:5], 4)}")
print(f"  Sum of weights:    {weights_port.sum():.6f}")
print(f"  HHI:               {diag['herfindahl_index']:.4f}")
print(f"  Effective bets:    {diag['effective_bets']:.2f}  (max possible: {N_port})")
print(f"  Diversification:   {diag['diversification_ratio']:.3f}")
print(f"  Entropy:           {diag['weight_entropy']:.3f}")

# ---------------------------------------------------------------------------
# 5. Alpha signal combination
# ---------------------------------------------------------------------------
print("\n--- 5. Alpha signal combination ---")

N_sig = 10
w_sig_true = rng.dirichlet(np.array([5.0] + [1.0] * (N_sig - 1)))   # dominant signal 0
X_sig = rng.randn(m, N_sig)
y_sig = X_sig @ w_sig_true + 0.05 * rng.randn(m)

combiner = AlphaSignalCombiner(lam=0.03, optimizer="mirror_prox", n_iters=600)
combiner.fit(X_sig, y_sig)
sig_w = combiner.get_signal_weights()
sig_diag = combiner.signal_diagnostics()

cosine_sig = float(np.dot(sig_w, w_sig_true) /
                   (np.linalg.norm(sig_w) * np.linalg.norm(w_sig_true)))
print(f"  Signal weights (first 4): {np.round(sig_w[:4], 4)}")
print(f"  Cosine similarity to truth: {cosine_sig:.4f}")
print(f"  Effective signals: {sig_diag['effective_signals']:.2f}")

# ---------------------------------------------------------------------------
# 6. Rolling backtest
# ---------------------------------------------------------------------------
print("\n--- 6. Rolling portfolio backtest ---")

T_bt, N_bt = 600, 8
returns_bt = rng.randn(T_bt, N_bt) * 0.01 + 0.0002

backtest = RollingPortfolioBacktest(
    window=120,
    refit_freq=20,
    lam=0.05,
    optimizer="mirror_descent",
    n_iters=300,
)
result = backtest.run(returns_bt)
summary = result.summary()

print(f"  Periods:       {summary['n_periods']}")
print(f"  Mean ENB:      {summary['mean_enb']:.2f}")
print(f"  Mean HHI:      {summary['mean_hhi']:.4f}")
print(f"  Mean turnover: {summary['mean_turnover']:.4f}")
print(f"  Total return:  {summary['total_return'] * 100:.2f}%")
if summary.get("sharpe") is not None:
    print(f"  Sharpe:        {summary['sharpe']:.3f}")
if summary.get("max_drawdown") is not None:
    print(f"  Max drawdown:  {summary['max_drawdown'] * 100:.2f}%")

# ---------------------------------------------------------------------------
# 7. Rolling signal backtest
# ---------------------------------------------------------------------------
print("\n--- 7. Rolling signal backtest ---")

backtest_sig = RollingSignalBacktest(
    window=120,
    refit_freq=20,
    lam=0.05,
    optimizer="mirror_descent",
    n_iters=300,
)
result_sig = backtest_sig.run(X_sig[:T_bt], y_sig[:T_bt])
summary_sig = result_sig.summary()
print(f"  Periods:       {summary_sig['n_periods']}")
print(f"  Mean OOS MSE:  {summary_sig.get('mean_oos_mse', float('nan')):.5f}")

# ---------------------------------------------------------------------------
# 8. Convergence analysis
# ---------------------------------------------------------------------------
print("\n--- 8. Convergence analysis ---")

T_th, G = 500, 1.0
eta_star = optimal_learning_rate(T_th, n, G)
kl_b = kl_regret_bound(T_th, n, G)
eu_b = euclidean_regret_bound(T_th, n, G)

print(f"  Optimal learning rate: {eta_star:.5f}")
print(f"  KL regret bound:       {kl_b:.4f}")
print(f"  Euclidean regret bound:{eu_b:.4f}")
print(f"  KL advantage:          {eu_b / kl_b:.2f}x")

model_conv = MirrorLinearRegression(lam=0.05, learning_rate=0.1, n_iters=500)
model_conv.fit(X, y)
print("\n  Convergence report:")
print_convergence_report(model_conv.loss_history, dim=n, eta=0.1, lam=0.05)

# ---------------------------------------------------------------------------
# 9. Riemannian geometry
# ---------------------------------------------------------------------------
print("\n--- 9. Riemannian geometry (Fisher-Rao) ---")

p = rng.dirichlet(np.ones(n))
q = rng.dirichlet(np.ones(n))

d_fr = fisher_rao_distance(p, q)
midpoint = geodesic(p, q, t=0.5)
ng = natural_gradient(np.ones(n), p)

print(f"  Fisher-Rao distance(p, q): {d_fr:.4f}")
print(f"  Geodesic midpoint sum:     {midpoint.sum():.6f}  (should be 1)")
print(f"  Midpoint all positive:     {(midpoint > 0).all()}")
print(f"  Natural gradient sum:      {ng.sum():.2e}  (should be ~0)")

# ---------------------------------------------------------------------------
# 10. Quick visualisation
# ---------------------------------------------------------------------------
model_lo = MirrorLinearRegression(lam=0.001, n_iters=500)
model_hi = MirrorLinearRegression(lam=0.30,  n_iters=500)
model_lo.fit(X, y)
model_hi.fit(X, y)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (label, w, color) in zip(axes, [
    ("true weights", w_true, "#555555"),
    (r"MIRAGE ($\lambda$=0.001)", model_lo.weights, "#d6604d"),
    (r"MIRAGE ($\lambda$=0.30)",  model_hi.weights, "#2166ac"),
]):
    ax.bar(range(n), w, color=color, alpha=0.85)
    ax.set_title(f"{label}\nH={entropy(w):.2f}  ENB={effective_number_of_bets(w):.1f}",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("feature index")
    ax.set_xticks([])

plt.suptitle("Weight distributions: low vs high entropy regularisation", fontweight="bold")
plt.tight_layout()
plt.savefig("demo_weights.png", bbox_inches="tight", dpi=120)
print("\nSaved demo_weights.png")

print("\n" + "=" * 60)
print("Demo complete.")
print("=" * 60)
