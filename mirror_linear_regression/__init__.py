"""
MIRAGE++: Mirror descent entropy-regularised linear regression
with Riemannian geometric extensions.

Quick start
-----------
    from mirror_linear_regression import MirrorLinearRegression

    model = MirrorLinearRegression(learning_rate=0.1, n_iters=500, lam=0.05)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.weights)          # simplex weights summing to 1
    print(model.convergence_summary())
"""

from .core import MirrorLinearRegression
from .loss import loss, loss_components, entropy
from .bregman import (
    BregmanDivergence,
    KLDivergence,
    SquaredEuclideanDivergence,
    ItakuraSaitoDivergence,
    BetaDivergence,
    get_divergence,
)
from .optim import (
    mirror_descent_step,
    mirror_prox_step,
    natural_gradient_descent_step,
    AdaMirror,
)
from .geometry import (
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
from .convergence import (
    kl_regret_bound,
    euclidean_regret_bound,
    optimal_learning_rate,
    convergence_rate_estimate,
    compute_regret,
    print_convergence_report,
)
from .portfolio_alloc import PortfolioAllocator
from .finance_alpha import AlphaSignalCombiner
from .backtest import RollingPortfolioBacktest, RollingSignalBacktest, BacktestResult
from .vector_space import EuclideanSpace, ProbabilitySimplexSpace, SPDManifoldSpace

__all__ = [
    # Core model
    "MirrorLinearRegression",
    # Loss
    "loss", "loss_components", "entropy",
    # Bregman divergences
    "BregmanDivergence", "KLDivergence", "SquaredEuclideanDivergence",
    "ItakuraSaitoDivergence", "BetaDivergence", "get_divergence",
    # Optimisers
    "mirror_descent_step", "mirror_prox_step",
    "natural_gradient_descent_step", "AdaMirror",
    # Geometry
    "natural_gradient", "natural_gradient_step",
    "fisher_rao_distance", "geodesic",
    "exponential_map", "logarithmic_map",
    "spd_riemannian_gradient", "spd_retraction",
    "spd_geodesic", "affine_invariant_distance", "project_to_spd",
    # Convergence
    "kl_regret_bound", "euclidean_regret_bound", "optimal_learning_rate",
    "convergence_rate_estimate", "compute_regret", "print_convergence_report",
    # Applications
    "PortfolioAllocator", "AlphaSignalCombiner",
    "RollingPortfolioBacktest", "RollingSignalBacktest", "BacktestResult",
    # Manifolds
    "EuclideanSpace", "ProbabilitySimplexSpace", "SPDManifoldSpace",
]
