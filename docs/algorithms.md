# Algorithms and Convergence Analysis

## 1. Mirror Descent (Exponentiated Gradient)

The canonical mirror descent update solves the proximal subproblem exactly in the geometry of a chosen Bregman divergence $D_\phi$:

$$\theta_{t+1} = \arg\min_{\theta \in \Delta} \left\{ \eta \langle \nabla\mathcal{L}(\theta_t),\, \theta \rangle + D_\phi(\theta \| \theta_t) \right\}$$

**With KL divergence** this has a closed form:

$$z_{t+1,i} = \log\theta_{t,i} + 1 - \eta\,\nabla_i\mathcal{L}(\theta_t)$$
$$\theta_{t+1,i} = \exp(z_{t+1,i} - 1) = \theta_{t,i}\,\exp(-\eta\,\nabla_i\mathcal{L}(\theta_t))$$
$$\theta_{t+1} \leftarrow \theta_{t+1} / \|\theta_{t+1}\|_1 \quad \text{(normalise)}$$

This is the **Hedge / Exponentiated Gradient** algorithm. The normalisation step replaces the explicit simplex projection.

```
Algorithm 1: Mirror Descent (KL)
Input: X, y, lambda, eta, T
Init:  theta_1 = (1/n) * ones(n)   [maximum entropy start]
For t = 1, ..., T:
    g  = (2/m) * X^T (X theta_t - y) + lambda * (1 + log(theta_t))
    theta_{t+1,i} = theta_{t,i} * exp(-eta * g_i)  for all i
    theta_{t+1}  = theta_{t+1} / sum(theta_{t+1})
    if |L(theta_{t+1}) - L(theta_t)| < tol: break
Return theta_T
```

**Computational cost per iteration:** $O(mn)$ — dominated by the matrix-vector product $X^\top(X\theta - y)$.

---

## 2. Natural Gradient Descent

The natural gradient corrects for the curvature of the Fisher-Rao metric. On the simplex the Fisher information matrix is $F(\theta) = \text{diag}(1/\theta)$, so the natural gradient is:

$$\tilde{\nabla}_i\mathcal{L} = \theta_i\!\left(\nabla_i\mathcal{L} - \sum_j \theta_j\,\nabla_j\mathcal{L}\right)$$

The update is a projected gradient step in the Euclidean sense but with the geometrically-corrected direction:

$$\theta_{t+1} = \text{proj}_\Delta\!\left(\theta_t - \eta\,\tilde{\nabla}\mathcal{L}(\theta_t)\right)$$

```
Algorithm 2: Natural Gradient Descent (Fisher-Rao)
Input: X, y, lambda, eta, T
Init:  theta_1 = (1/n) * ones(n)
For t = 1, ..., T:
    g      = (2/m) * X^T (X theta_t - y) + lambda * (1 + log(theta_t))
    center = sum_i theta_{t,i} * g_i          [theta^T g]
    ng_i   = theta_{t,i} * (g_i - center)     [natural gradient]
    theta_{t+1} = proj_simplex(theta_t - eta * ng)
    if convergence: break
Return theta_T
```

**Note:** At small $\eta$, NGD and KL mirror descent agree to first order ($O(\eta)$). They diverge at $O(\eta^2)$: NGD uses second-order Fisher information; KL-MD uses the exact Bregman proximal step.

---

## 3. Mirror Prox (Extragradient)

Mirror Prox (Nemirovski, 2004) evaluates the gradient at a **half-step** $\theta_{t+1/2}$ before computing the full update. This removes the oscillation that limits standard mirror descent on smooth objectives, yielding $O(1/T)$ convergence instead of $O(1/\sqrt{T})$.

```
Algorithm 3: Mirror Prox
Input: X, y, lambda, eta, T
Init:  theta_1 = (1/n) * ones(n)
For t = 1, ..., T:
    g_half     = gradient(theta_t)
    theta_half = mirror_step(g_half, theta_t, eta)    [half-step]
    g_full     = gradient(theta_half)
    theta_{t+1} = mirror_step(g_full, theta_t, eta)   [full step from theta_t]
    if convergence: break
Return theta_T
```

The key insight: the **full step uses the gradient at the half-step** but is **centred at $\theta_t$** (not $\theta_{t+1/2}$). This is the extragradient property that provides the improved rate.

**Convergence:** For $\beta$-smooth objectives and $\eta \leq 1/\beta$:

$$\min_{t \leq T} \|\nabla\mathcal{L}(\theta_t)\| \leq \frac{2D_\phi(\theta^*, \theta_1)}{\eta T}$$

---

## 4. AdaMirror (Adaptive Mirror Descent)

AdaMirror accumulates per-coordinate squared gradient norms and uses them to compute adaptive step sizes:

$$G_{t,i} = \sum_{s=1}^t g_{s,i}^2, \qquad \hat{\eta}_{t,i} = \frac{\eta}{\sqrt{G_{t,i}} + \epsilon}$$

The update then applies KL mirror descent with coordinate-wise rates:

$$\theta_{t+1,i} \propto \theta_{t,i} \cdot \exp(-\hat{\eta}_{t,i}\,g_{t,i})$$

```
Algorithm 4: AdaMirror
Input: X, y, lambda, eta, eps=1e-8, T
Init:  theta_1 = (1/n) * ones(n),  G = zeros(n)
For t = 1, ..., T:
    g        = gradient(theta_t)
    G        = G + g^2                         [accumulate sq. gradients]
    eta_hat  = eta / (sqrt(G) + eps)           [per-coordinate rates]
    theta_{t+1,i} = theta_{t,i} * exp(-eta_hat_i * g_i)
    theta_{t+1}   = theta_{t+1} / sum(theta_{t+1})
    if convergence: break
Return theta_T
```

AdaMirror is particularly effective when gradient magnitudes vary widely across coordinates — common when signals have different scales or volatilities.

---

## 5. Convergence Theory

### Theorem 1: KL Regret Bound

Let $\{\theta_t\}$ be the sequence produced by KL mirror descent with step size $\eta > 0$ and gradient bound $\|\nabla\mathcal{L}(\theta_t)\|_\infty \leq G$. For any comparator $\theta^* \in \Delta_{n-1}$:

$$\sum_{t=1}^T \left[\mathcal{L}(\theta_t) - \mathcal{L}(\theta^*)\right] \leq \frac{D_\text{KL}(\theta^* \| \theta_1)}{\eta} + \frac{\eta G^2 T}{2}$$

Setting $\eta^* = \sqrt{2 D_\text{KL}(\theta^* \| \theta_1) / (G^2 T)}$ and initialising at $\theta_1 = \mathbf{1}/n$ (so $D_\text{KL}(\theta^* \| \theta_1) \leq \log n$):

$$\text{Regret}_T \leq G\sqrt{2T\log n}$$

### Theorem 2: Dimensional Advantage

Comparing KL and Euclidean mirror descent at their respective optimal step sizes:

| Method | Regret Bound | Optimal $\eta$ |
|--------|-------------|----------------|
| KL mirror descent | $G\sqrt{2T\log n}$ | $\sqrt{2\log n / (TG^2)}$ |
| Euclidean (projected GD) | $G\sqrt{2nT}$ | $\sqrt{2 / (nTG^2)}$ |
| **Ratio** | $\sqrt{n/\log n}$ | — |

For $n = 1000$, the KL bound is $\approx 10\times$ smaller than Euclidean. The advantage grows unboundedly: $\sqrt{n/\log n} \to \infty$ as $n \to \infty$.

### Theorem 3: Linear Convergence with Strong Convexity

When $\lambda > 0$, the entropy term $-\lambda H(\theta)$ makes $\mathcal{L}$ **strongly convex** with modulus $\mu = \lambda/\max_i\theta_i$. KL mirror descent then converges at a **linear rate**:

$$\mathcal{L}(\theta_t) - \mathcal{L}(\theta^*) \leq e^{-\mu\eta t}\left(\mathcal{L}(\theta_1) - \mathcal{L}(\theta^*)\right)$$

This transitions from the $O(1/\sqrt{T})$ sublinear rate (when $\lambda = 0$) to a linear rate (when $\lambda > 0$). Higher $\lambda$ accelerates convergence but at the cost of a biased solution.

### Theorem 4: Minimax Lower Bound — Optimality of KL Mirror Descent

**Statement.** For any deterministic online algorithm $\text{ALG}$ on $\Delta_{n-1}$ with gradient bound $\|g_t\|_\infty \leq G$, there exists an adversarial sequence of convex losses such that:

$$\text{Regret}_T(\text{ALG}) \geq \frac{G}{2}\sqrt{\frac{T \log n}{2}} = \Omega\!\left(G\sqrt{T \log n}\right)$$

**Proof sketch** (Cesa-Bianchi & Lugosi, 2006, Thm 3.4).
Consider the symmetric adversary: at each step $t$, independently for each coordinate $i$, set $g_{t,i} = +G$ or $-G$ with equal probability. For any *deterministic* algorithm, the expected regret against the best fixed arm $\theta^* = e_{i^*}$ is:

$$\mathbb{E}[\text{Regret}_T] = \mathbb{E}\!\left[\sum_t \langle g_t, \theta_t - e_{i^*}\rangle\right] \geq \frac{G}{2}\sqrt{\frac{T \log n}{2}}$$

by a birthday-paradox counting argument: with $n$ coordinates and $T$ rounds, at least $\Omega(\log n)$ coordinates will exhibit persistent bias that any algorithm must incur.

**Corollary: KL mirror descent is minimax optimal to within a constant factor.**

| Method | Regret Bound | Factor vs Lower Bound |
| --- | --- | --- |
| KL Mirror Descent | $G\sqrt{2T\log n}$ | $4$ |
| Minimax Lower Bound | $\frac{G}{2}\sqrt{\frac{T\log n}{2}}$ | $1$ (by definition) |
| Euclidean (PGD) | $G\sqrt{2nT}$ | $\sqrt{n/\log n} \to \infty$ |

The gap between KL and the lower bound is a constant factor of exactly $4$ (independent of $n$ and $T$) — KL mirror descent is minimax optimal in rate. The gap between Euclidean and the lower bound is $\sqrt{n/\log n}$, which grows without bound. This is a **fundamental, information-theoretic separation** — not an artefact of analysis — proving that Euclidean geometry is the wrong choice for the simplex at any scale.

**Computational consequence.** The lower bound applies to *any* algorithm, including Frank-Wolfe and Riemannian SGD without entropy regularisation. MIRAGE++ with KL geometry and entropy regularisation is simultaneously:

1. **Minimax optimal** (up to constant 2) in regret
2. **Linearly convergent** (from strong convexity)
3. **Interior to the simplex** at all iterates (from the log-barrier)

No other simplex-constrained algorithm achieves all three simultaneously.

---

## 6. Initialisation Strategy

All algorithms initialise at the **maximum-entropy point**:

$$\theta_1 = \frac{\mathbf{1}}{n} = (1/n, \ldots, 1/n)$$

This choice:
1. Minimises the initial KL divergence to any comparator: $D_\text{KL}(\theta^* \| \theta_1) \leq \log n$
2. Provides the tightest regret bound from Theorem 1
3. Is the least committal starting point — no prior information is assumed

---

## 7. Stopping Criterion

Training stops when the absolute improvement in loss falls below tolerance $\tau$:

$$|\mathcal{L}(\theta_{t+1}) - \mathcal{L}(\theta_t)| < \tau$$

Default $\tau = 10^{-7}$. Convergence typically occurs well within `n_iters` (default 500) for well-scaled data. For ill-conditioned problems (high correlation, fat tails), increase `n_iters` or lower `learning_rate`.
