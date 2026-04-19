"""
Convergence analysis, regret bounds, and diagnostics for MIRAGE++.

Theoretical guarantees
----------------------

All bounds assume convex loss functions L_t, feasible set Delta_{n-1},
initial point theta_1 = uniform (1/n, ..., 1/n), and gradient bound
||nabla L_t||_inf <= G.

Theorem 1 (KL Mirror Descent Regret Bound).
    Let theta_1, ..., theta_T be produced by KL mirror descent with
    constant step size eta.  For any theta* in Delta_{n-1}:

        sum_{t=1}^{T} L_t(theta_t) - sum_{t=1}^{T} L_t(theta*)
            <= D_KL(theta* || theta_1) / eta  +  eta * T * G^2 / 2
            <= log(n) / eta  +  eta * T * G^2 / 2

    Choosing eta* = sqrt(2 log(n) / (T G^2)) gives:

        Regret_T  <=  G * sqrt(2 T log(n))

Theorem 2 (KL vs Euclidean).
    Projected gradient descent on Delta_{n-1} has regret bound:

        Regret_T (Euclidean)  <=  sqrt(2n) * G * sqrt(T)

    KL mirror descent improves the dimension dependence from sqrt(n) to
    sqrt(log n) -- exponentially better for large n.

Theorem 3 (Mirror-Prox, Smooth Objectives).
    For L-smooth objectives, Mirror-Prox achieves:

        L(theta_avg) - L(theta*)  <=  2 L * D_KL(theta*||theta_1) / T
                                    = O(L log(n) / T)

    vs O(L log(n) / sqrt(T)) for standard mirror descent.

Theorem 4 (Entropy Regularisation and Strong Convexity).
    With entropy regularisation strength lambda > 0, the MIRAGE++ objective:

        F(theta) = MSE(theta) - lambda * H(theta)

    is lambda-strongly convex w.r.t. the KL Bregman divergence on Delta_{n-1}
    (since -H is the generator of KL divergence, hence strongly convex).
    Under mu-strong convexity with constant step size eta:

        F(theta_T) - F(theta*)  <=  exp(-mu * eta * T) * (F(theta_1) - F(theta*))

    i.e. exponential (linear) convergence vs the polynomial rate without
    strong convexity.

References:
  Shalev-Shwartz (2012). Online Learning and Online Convex Optimization.
  Bubeck (2015). Convex Optimization: Algorithms and Complexity.
  Nesterov (2009). Primal-dual subgradient methods for convex problems.
  Hazan (2016). Introduction to Online Convex Optimization.
"""

import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# Theoretical bounds
# ---------------------------------------------------------------------------

def kl_regret_bound(
    num_steps: int,
    dim: int,
    gradient_bound: float = 1.0,
    eta: Optional[float] = None,
) -> float:
    """
    Upper bound on the KL mirror descent regret (Theorem 1).

        Regret_T <= log(n) / eta  +  eta * T * G^2 / 2

    With the optimal eta = sqrt(2 log(n) / (T G^2)):

        Regret_T <= G * sqrt(2 T log(n))

    Args:
        num_steps: Number of optimisation steps T.
        dim: Simplex dimension n.
        gradient_bound: Bound G on ||nabla L_t||_inf.
        eta: Learning rate (if None, uses the optimal eta).

    Returns:
        Regret upper bound.
    """
    if eta is None:
        return gradient_bound * np.sqrt(2.0 * num_steps * np.log(dim))
    return (
        np.log(dim) / eta
        + eta * num_steps * gradient_bound ** 2 / 2.0
    )


def euclidean_regret_bound(
    num_steps: int,
    dim: int,
    gradient_bound: float = 1.0,
) -> float:
    """
    Upper bound on the projected gradient descent regret (Theorem 2).

        Regret_T <= sqrt(2n) * G * sqrt(T)

    Args:
        num_steps: Number of steps T.
        dim: Simplex dimension n.
        gradient_bound: Bound G on ||nabla L_t||_2.

    Returns:
        Regret upper bound.
    """
    return np.sqrt(2.0 * dim) * gradient_bound * np.sqrt(num_steps)


def optimal_learning_rate(
    num_steps: int,
    dim: int,
    gradient_bound: float = 1.0,
) -> float:
    """
    Minimax-optimal constant learning rate for KL mirror descent over T steps:

        eta* = sqrt(2 log(n) / (T G^2))

    This minimises the regret bound in Theorem 1 over all constant eta.

    Args:
        num_steps: T (must be known in advance for the offline setting).
        dim: Simplex dimension n.
        gradient_bound: Bound G on ||nabla L_t||_inf.

    Returns:
        Optimal step size eta*.
    """
    return np.sqrt(2.0 * np.log(dim) / (num_steps * gradient_bound ** 2))


def strong_convexity_trajectory(
    initial_gap: float,
    mu: float,
    eta: float,
    num_steps: int,
) -> np.ndarray:
    """
    Theoretical loss-gap trajectory under mu-strong convexity (Theorem 4):

        F(theta_t) - F(theta*)  <=  gap_0 * exp(-mu * eta * t)

    With lambda-entropy regularisation, mu >= lambda.

    Args:
        initial_gap: F(theta_0) - F(theta*) at initialisation.
        mu: Strong convexity constant (>= lambda, the regularisation strength).
        eta: Learning rate.
        num_steps: Number of steps T.

    Returns:
        Array of upper bounds on (F_t - F*), shape (T,).
    """
    t = np.arange(num_steps)
    return initial_gap * np.exp(-mu * eta * t)


# ---------------------------------------------------------------------------
# Empirical diagnostics
# ---------------------------------------------------------------------------

def compute_regret(
    loss_history: List[float],
    optimal_loss: float,
) -> np.ndarray:
    """
    Compute cumulative regret from an observed loss trajectory:

        Regret_T = sum_{t=1}^{T} (L_t - L*)

    Args:
        loss_history: Per-step losses L_1, ..., L_T.
        optimal_loss: Offline optimal loss L* (use min(loss_history) as proxy).

    Returns:
        Array of cumulative regrets, shape (T,).
    """
    losses = np.array(loss_history)
    return np.cumsum(losses - optimal_loss)


def convergence_rate_estimate(loss_history: List[float]) -> float:
    """
    Estimate the empirical convergence rate from a loss trajectory.

    Fits the log-linear model  log L_t = b - alpha * log t  by least squares
    and returns the exponent alpha.  Larger alpha means faster convergence
    (L_t ~ C * t^{-alpha}).

    Args:
        loss_history: List of per-step losses.

    Returns:
        Estimated convergence rate alpha (float, or nan if insufficient data).
    """
    losses = np.array(loss_history, dtype=float)
    losses = losses[losses > 0]
    if len(losses) < 10:
        return float("nan")
    t = np.arange(1, len(losses) + 1, dtype=float)
    log_t = np.log(t)
    log_l = np.log(losses)
    mat = np.column_stack([np.ones_like(log_t), log_t])
    coeffs, *_ = np.linalg.lstsq(mat, log_l, rcond=None)
    return float(-coeffs[1])


def effective_dimension(loss_history: List[float]) -> float:
    """
    Estimate the effective problem dimension from the empirical regret trajectory.

    Fits  Regret_T ~ sqrt(d_eff * T)  to the observed cumulative regret and
    returns d_eff.  Useful for diagnosing whether the KL geometry provides
    a dimension reduction relative to Euclidean mirror descent.

    Args:
        loss_history: Per-step losses.

    Returns:
        Estimated effective dimension d_eff.
    """
    losses = np.array(loss_history, dtype=float)
    opt = np.min(losses)
    regret = np.cumsum(losses - opt)
    t = np.arange(1, len(regret) + 1, dtype=float)
    # Regret_T ~ C * sqrt(T)  =>  Regret_T^2 / T ~ C^2 = d_eff (up to constants)
    ratios = regret ** 2 / t
    return float(np.median(ratios[len(ratios) // 2:]))  # use second half (stable)


def print_convergence_report(
    loss_history: List[float],
    dim: int,
    eta: float,
    gradient_bound: float = 1.0,
    lam: float = 0.0,
) -> None:
    """
    Print a formatted convergence report comparing empirical behaviour
    against theoretical bounds.

    Args:
        loss_history: Per-step losses from training.
        dim: Simplex dimension n.
        eta: Learning rate used.
        gradient_bound: Assumed bound G on ||nabla L||_inf.
        lam: Entropy regularisation strength (for strong-convexity bound).
    """
    num_steps = len(loss_history)
    rate = convergence_rate_estimate(loss_history)
    kl_bound = kl_regret_bound(num_steps, dim, gradient_bound)
    eu_bound = euclidean_regret_bound(num_steps, dim, gradient_bound)
    opt_eta = optimal_learning_rate(num_steps, dim, gradient_bound)

    print("=" * 60)
    print("  MIRAGE++ Convergence Report")
    print("=" * 60)
    print(f"  Iterations run:       {num_steps}")
    print(f"  Simplex dimension:    {dim}")
    print(f"  Learning rate (used): {eta:.6f}")
    print(f"  Optimal eta* (KL):    {opt_eta:.6f}")
    print()
    print(f"  Initial loss:         {loss_history[0]:.6f}")
    print(f"  Final loss:           {loss_history[-1]:.6f}")
    print(f"  Loss reduction:       {loss_history[0] - loss_history[-1]:.6f}")
    print(f"  Empirical rate alpha: {rate:.3f}  (L_t ~ t^(-alpha))")
    print()
    print(f"  KL regret bound:      {kl_bound:.4f}  [O(sqrt(T log n))]")
    print(f"  Euclidean bound:      {eu_bound:.4f}  [O(sqrt(nT))]")
    print(f"  KL / Euclidean:       {kl_bound / eu_bound:.4f}  (<1 means KL wins)")
    if lam > 0:
        t_to_eps = int(np.ceil(1.0 / (lam * eta)))
        print(f"  lambda={lam:.4f}: linear convergence, ~{t_to_eps} iters to 1/e gap")
    print("=" * 60)
