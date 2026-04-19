"""Generate all publication-quality figures for MIRAGE++."""
import sys, warnings, copy
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from mirror_linear_regression import MirrorLinearRegression
from mirror_linear_regression.convergence import (
    kl_regret_bound, euclidean_regret_bound, strong_convexity_trajectory,
)
from mirror_linear_regression.utils_math import entropy, effective_number_of_bets
from mirror_linear_regression.bregman import (
    SquaredEuclideanDivergence, ItakuraSaitoDivergence, BetaDivergence,
)
from mirror_linear_regression.geometry import geodesic, fisher_rao_distance

import pathlib
pathlib.Path("visuals").mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 11, "axes.titlesize": 12,
})

rng = np.random.RandomState(42)
print("Generating visuals...")

# ------------------------------------------------------------------
# Shared dataset
# ------------------------------------------------------------------
n, m = 20, 300
X = rng.randn(m, n)
w_true = rng.dirichlet(np.ones(n))
y = X @ w_true + 0.01 * rng.randn(m)

# ------------------------------------------------------------------
# 1. KL vs Euclidean regret bound scaling
# ------------------------------------------------------------------
dims = [5, 10, 20, 50, 100, 200, 500, 1000]
T_r, G = 1000, 1.0
kl_b = [kl_regret_bound(T_r, nd, G) for nd in dims]
eu_b = [euclidean_regret_bound(T_r, nd, G) for nd in dims]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(dims, kl_b, "o-", color="#2166ac", lw=2, ms=6,
          label=r"KL (MIRAGE++):  $O(\sqrt{T \log n})$")
ax.loglog(dims, eu_b, "s-", color="#d6604d", lw=2, ms=6,
          label=r"Euclidean (Ridge):  $O(\sqrt{nT})$")
ax.fill_between(dims, kl_b, eu_b, alpha=0.12, color="#2166ac")
ax.set_xlabel("Dimension $n$"); ax.set_ylabel("Regret bound")
ax.set_title("KL vs Euclidean Regret Bound Scaling", fontweight="bold")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("visuals/kl_vs_euclidean_regret.png", bbox_inches="tight")
plt.close()
print("  1/7  kl_vs_euclidean_regret.png")

# ------------------------------------------------------------------
# 2. Convergence curves: all 4 optimisers
# ------------------------------------------------------------------
OPT_CFG = [
    ("mirror_descent",   0.10, "#2166ac"),
    ("natural_gradient", 0.05, "#1a9850"),
    ("mirror_prox",      0.10, "#d6604d"),
    ("ada_mirror",       0.20, "#762a83"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for opt, lr, color in OPT_CFG:
    mo = MirrorLinearRegression(optimizer=opt, lam=0.01, learning_rate=lr,
                                n_iters=600, tol=0)
    mo.fit(X, y)
    label = opt.replace("_", " ").title()
    ax1.semilogy(mo.loss_history,    color=color, lw=1.8, label=label)
    ax2.plot(mo.entropy_history, color=color, lw=1.8, label=label)

ax1.set_xlabel("Iteration"); ax1.set_ylabel("Loss (log scale)")
ax1.set_title("Loss Convergence", fontweight="bold")
ax1.legend(frameon=False, fontsize=9)
ax2.set_xlabel("Iteration"); ax2.set_ylabel(r"Entropy $H(\theta)$")
ax2.set_title("Weight Entropy During Training", fontweight="bold")
ax2.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("visuals/convergence_curves.png", bbox_inches="tight")
plt.close()
print("  2/7  convergence_curves.png")

# ------------------------------------------------------------------
# 3. Lambda sensitivity: MSE / H(theta) / ENB
# ------------------------------------------------------------------
lambdas = np.logspace(-3, 0, 25)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
mse_v, H_v, enb_v = [], [], []
for lam in lambdas:
    fold_mse = []
    for tr, te in kf.split(X):
        mc = copy.deepcopy(MirrorLinearRegression(lam=lam, n_iters=400, tol=1e-7))
        mc.fit(X[tr], y[tr])
        fold_mse.append(mean_squared_error(y[te], mc.predict(X[te])))
    mf = MirrorLinearRegression(lam=lam, n_iters=400, tol=1e-7)
    mf.fit(X, y)
    mse_v.append(float(np.mean(fold_mse)))
    H_v.append(float(entropy(mf.weights)))
    enb_v.append(float(effective_number_of_bets(mf.weights)))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].semilogx(lambdas, mse_v, color="#2166ac", lw=2)
axes[0].set_xlabel(r"$\lambda$"); axes[0].set_ylabel("CV MSE")
axes[0].set_title(r"CV MSE vs $\lambda$", fontweight="bold")
axes[1].semilogx(lambdas, H_v, color="#d6604d", lw=2)
axes[1].set_xlabel(r"$\lambda$"); axes[1].set_ylabel(r"Entropy $H(\theta)$")
axes[1].set_title(r"Weight Entropy vs $\lambda$", fontweight="bold")
axes[2].semilogx(lambdas, enb_v, color="#1a9850", lw=2)
axes[2].axhline(n, ls="--", color="gray", lw=1, label=f"max = {n}")
axes[2].set_xlabel(r"$\lambda$"); axes[2].set_ylabel("ENB")
axes[2].set_title(r"Effective Number of Bets vs $\lambda$", fontweight="bold")
axes[2].legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("visuals/lambda_sensitivity.png", bbox_inches="tight")
plt.close()
print("  3/7  lambda_sensitivity.png")

# ------------------------------------------------------------------
# 4. Weight profiles: OLS vs MIRAGE++ variants
# ------------------------------------------------------------------
w_ols = np.clip(LinearRegression(fit_intercept=False).fit(X, y).coef_, 0, None)
w_ols /= w_ols.sum() + 1e-12
w_ridge = np.clip(Ridge(alpha=1.0, fit_intercept=False).fit(X, y).coef_, 0, None)
w_ridge /= w_ridge.sum() + 1e-12
weights_wp = {
    "OLS (projected)": w_ols,
    "Ridge (projected)": w_ridge,
}
for lam, label in [(0.001, "MIRAGE (lam=0.001)"),
                   (0.05,  "MIRAGE (lam=0.05)"),
                   (0.30,  "MIRAGE (lam=0.30)")]:
    m5 = MirrorLinearRegression(lam=lam, n_iters=600, tol=1e-7)
    m5.fit(X, y)
    weights_wp[label] = m5.weights

palette = ["#d6604d", "#f4a582", "#2166ac", "#4393c3", "#053061"]
fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
for ax, (label, w), color in zip(axes, weights_wp.items(), palette):
    ax.bar(range(len(w)), w, color=color, alpha=0.85, width=0.8)
    ax.plot(range(len(w_true)), w_true, "k--", lw=1.2, alpha=0.5, label="true")
    ax.set_title(f"{label}\nH={entropy(w):.2f}  ENB={effective_number_of_bets(w):.1f}",
                 fontsize=8, fontweight="bold")
    ax.set_xlabel("feature index"); ax.set_xticks([])
    if ax is axes[0]:
        ax.set_ylabel("weight")
    ax.legend(fontsize=7, frameon=False)
plt.suptitle("Weight Profiles: Projection vs Entropy Regularisation", fontweight="bold")
plt.tight_layout()
plt.savefig("visuals/weight_profiles.png", bbox_inches="tight")
plt.close()
print("  4/7  weight_profiles.png")

# ------------------------------------------------------------------
# 5. Bregman divergence comparison
# ------------------------------------------------------------------
n2, m2 = 10, 200
X2 = rng.randn(m2, n2)
w2 = rng.dirichlet(np.ones(n2))
y2 = X2 @ w2 + 0.02 * rng.randn(m2)

div_cfg = [
    ("KL (default)",     None,                          "#2166ac"),
    ("Euclidean",        SquaredEuclideanDivergence(),   "#d6604d"),
    ("Itakura-Saito",    ItakuraSaitoDivergence(),       "#1a9850"),
    ("Beta (b=1.5)",     BetaDivergence(1.5),            "#762a83"),
]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (label, div, color) in zip(axes, div_cfg):
    m6 = MirrorLinearRegression(lam=0.02, n_iters=500, tol=0, divergence=div)
    m6.fit(X2, y2)
    ax.semilogy(m6.loss_history, lw=1.8, color=color)
    ax.set_title(f"{label}\nfinal H={entropy(m6.weights):.2f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("iteration")
    if ax is axes[0]:
        ax.set_ylabel("loss (log scale)")
plt.suptitle("Loss Convergence by Bregman Divergence", fontweight="bold")
plt.tight_layout()
plt.savefig("visuals/bregman_divergence_comparison.png", bbox_inches="tight")
plt.close()
print("  5/7  bregman_divergence_comparison.png")

# ------------------------------------------------------------------
# 6. Strong-convexity linear convergence
# ------------------------------------------------------------------
T_conv = 300
configs = [
    (0.5, "--", "#d6604d", r"$\mu=0.5$ (weak regularisation)"),
    (1.0, "-",  "#2166ac", r"$\mu=1.0$ (moderate)"),
    (2.0, "-",  "#1a9850", r"$\mu=2.0$ (strong regularisation)"),
]
fig, ax = plt.subplots(figsize=(7, 4.5))
for mu, ls, color, label in configs:
    traj = strong_convexity_trajectory(1.0, mu=mu, eta=0.1, num_steps=T_conv)
    ax.semilogy(traj, ls=ls, color=color, lw=2, label=label)
ax.set_xlabel("Iteration $t$")
ax.set_ylabel(r"$\mathcal{L}_t - \mathcal{L}^*$")
ax.set_title(r"Linear Convergence Rate: $\mathcal{L}_t - \mathcal{L}^* \leq "
             r"e^{-\mu\eta t}(\mathcal{L}_1 - \mathcal{L}^*)$", fontweight="bold")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("visuals/linear_convergence_theory.png", bbox_inches="tight")
plt.close()
print("  6/7  linear_convergence_theory.png")

# ------------------------------------------------------------------
# 7. Fisher-Rao geodesic on the simplex
# ------------------------------------------------------------------
p = np.array([0.7, 0.2, 0.1])
q = np.array([0.1, 0.3, 0.6])
ts = np.linspace(0, 1, 80)
path = np.array([geodesic(p, q, t) for t in ts])
dist = fisher_rao_distance(p, q)

fig, ax = plt.subplots(figsize=(7, 4.5))
colors_geo = ["#2166ac", "#d6604d", "#1a9850"]
labels_geo = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
for i, (color, lbl) in enumerate(zip(colors_geo, labels_geo)):
    ax.plot(ts, path[:, i], color=color, lw=2.2, label=lbl)
    ax.scatter([0, 1], [p[i], q[i]], color=color, s=70, zorder=5)
ax.axvline(0, color="gray", lw=0.8, ls=":"); ax.axvline(1, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Geodesic parameter $t$")
ax.set_ylabel("Component weight")
ax.set_title(f"Fisher-Rao Geodesic on the Probability Simplex\n"
             f"(d_FR(p, q) = {dist:.3f})", fontweight="bold")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("visuals/fisher_rao_geodesic.png", bbox_inches="tight")
plt.close()
print("  7/7  fisher_rao_geodesic.png")

print("\nAll visuals saved to visuals/")
