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

Theorem 5 (Minimax Lower Bound — Optimality of KL Mirror Descent).
    For any deterministic online algorithm ALG operating on the simplex
    Delta_{n-1} with gradient bound ||g_t||_inf <= G, there exists an
    adversarial sequence of convex losses such that:

        Regret_T(ALG)  >=  (G / 2) * sqrt(T * log(n) / 2)
                        =  Omega(G * sqrt(T log n))

    Proof sketch (Cesa-Bianchi & Lugosi, 2006, Thm 3.4):
    Consider the n-dimensional symmetric random walk: at each step t,
    the adversary picks loss l_t(theta) = <g_t, theta> where g_t is
    chosen from {-G * e_i, +G * e_i} for each coordinate i independently.
    For any deterministic algorithm, the expected regret against the best
    fixed action is at least (G/2)*sqrt(T*log(n)/2) by a birthday-paradox
    counting argument over the 2^n possible loss sequences.

    Consequence: KL mirror descent with eta* = sqrt(2 log(n) / (T G^2))
    achieves Regret_T <= G*sqrt(2T log n), matching the lower bound up to
    a constant factor of 4:

        KL upper bound / minimax lower bound
            = G*sqrt(2T log n) / ((G/2)*sqrt(T log n / 2))
            = 4

    It is *minimax optimal to within a constant factor of 4*.  The order
    O(sqrt(T log n)) cannot be improved — no algorithm can achieve
    o(sqrt(T log n)) regret on the simplex in the worst case.

    The Euclidean bound O(sqrt(nT)) is suboptimal by a factor of
    sqrt(n/log n) relative to the lower bound, which grows without bound.
    This is a fundamental, information-theoretic separation.

References:
  Shalev-Shwartz (2012). Online Learning and Online Convex Optimization.
  Bubeck (2015). Convex Optimization: Algorithms and Complexity.
  Nesterov (2009). Primal-dual subgradient methods for convex problems.
  Hazan (2016). Introduction to Online Convex Optimization.
  Cesa-Bianchi & Lugosi (2006). Prediction, Learning, and Games. CUP.
"""

from typing import List, Optional

import numpy as np


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


def minimax_lower_bound(
    num_steps: int,
    dim: int,
    gradient_bound: float = 1.0,
) -> float:
    """
    Minimax lower bound on regret for any algorithm on the simplex (Theorem 5).

    For any deterministic online algorithm and any gradient bound G, there
    exists an adversarial loss sequence such that:

        Regret_T  >=  (G / 2) * sqrt(T * log(n) / 2)

    This establishes that KL mirror descent is minimax optimal to within
    a constant factor: its regret G*sqrt(2T log n) matches this lower
    bound up to a factor of 2.

    Args:
        num_steps: T
        dim: Simplex dimension n.
        gradient_bound: G

    Returns:
        Lower bound on worst-case regret.
    """
    return (gradient_bound / 2.0) * np.sqrt(num_steps * np.log(dim) / 2.0)


def kl_optimality_gap(
    num_steps: int,
    dim: int,
    gradient_bound: float = 1.0,
) -> float:
    """
    Ratio of the KL upper bound to the minimax lower bound.

    A ratio of exactly 1 would mean KL mirror descent is perfectly minimax
    optimal.  The actual ratio is 4, reflecting that the KL upper bound
    G*sqrt(2T log n) exceeds the lower bound (G/2)*sqrt(T log n/2) by a
    factor of 4.  This constant gap does not grow with n or T — KL mirror
    descent is minimax optimal in rate, to within a constant factor of 4.

    Args:
        num_steps: T
        dim: n
        gradient_bound: G

    Returns:
        Ratio: kl_upper_bound / minimax_lower_bound (should be ~2.0).
    """
    upper = kl_regret_bound(num_steps, dim, gradient_bound)
    lower = minimax_lower_bound(num_steps, dim, gradient_bound)
    return float(upper / lower) if lower > 0 else float("inf")


def euclidean_suboptimality_factor(dim: int) -> float:
    """
    The factor by which Euclidean mirror descent is worse than minimax optimal.

    The Euclidean regret bound is O(sqrt(nT)) while the minimax lower bound
    is Omega(sqrt(T log n)).  The ratio is:

        sqrt(nT) / sqrt(T log n)  =  sqrt(n / log n)

    This factor grows without bound as n increases, demonstrating that
    using Euclidean geometry on the simplex is not merely a constant
    suboptimal — it is fundamentally the wrong geometry.

    Args:
        dim: Simplex dimension n.

    Returns:
        sqrt(n / log(n)), the asymptotic suboptimality factor.
    """
    return float(np.sqrt(dim / np.log(dim))) if dim > 1 else 1.0


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
    lower = minimax_lower_bound(num_steps, dim, gradient_bound)
    sub_factor = euclidean_suboptimality_factor(dim)
    print(f"  KL regret bound:      {kl_bound:.4f}  [O(sqrt(T log n))]")
    print(f"  Minimax lower bound:  {lower:.4f}  [Omega(sqrt(T log n))]  (Thm 5)")
    print(f"  KL / lower bound:     {kl_bound / lower:.4f}  (constant factor = 4.0)")
    print(f"  Euclidean bound:      {eu_bound:.4f}  [O(sqrt(nT))]")
    print(f"  Euclidean subopt.:    {sub_factor:.2f}x  [sqrt(n/log n)]")
    if lam > 0:
        t_to_eps = int(np.ceil(1.0 / (lam * eta)))
        print(f"  lambda={lam:.4f}: linear convergence, ~{t_to_eps} iters to 1/e gap")
    print("=" * 60)
