# Experiment Methodology

## Overview

MIRAGE++ is evaluated across three complementary experiment types:

| Script | Focus | Datasets | Models |
|--------|-------|----------|--------|
| `benchmark.py` | Predictive accuracy vs baselines | 6 synthetic scenarios | 15 models (7 sklearn + 8 MIRAGE++) |
| `ablation.py` | Component contribution analysis | Synthetic (controlled) | MIRAGE++ variants |
| `examples/market_experiments.py` | Real-world validation | 4 market datasets | MIRAGE++ + sklearn |

---

## 1. Benchmark (`benchmark.py`)

### Datasets

| Name | $m$ | $n$ | Ground truth | Focus |
|------|-----|-----|-------------|-------|
| Alpha signals | 300 | 10 | Dirichlet weights | Weight recovery |
| Portfolio (20 assets) | 200 | 20 | None | Diversification |
| Volatility (8 features) | 300 | 8 | Positive coefs | Constrained regression |
| Ensemble (15 models) | 300 | 15 | Dirichlet weights | Combination |
| Macro (12 features) | 300 | 12 | Mixed-sign coefs | Constrained approximation |
| HighDim (50 signals) | 500 | 50 | Dirichlet weights | Scalability |

### Models

**Baselines (sklearn, weights projected to simplex post-hoc):**
- OLS: unconstrained least squares
- Ridge($\alpha$): L2 regularisation, $\alpha \in \{0.001, 0.1, 1.0\}$
- Lasso($\alpha$): L1 regularisation, $\alpha \in \{0.001, 0.01\}$
- ElasticNet: combined L1+L2

**MIRAGE++ variants:**
- MIRAGE-MD($\lambda$): mirror descent, $\lambda \in \{0.001, 0.01, 0.05, 0.1\}$
- MIRAGE-NGD: natural gradient descent, $\lambda=0.01$
- MIRAGE-MP: mirror prox, $\lambda=0.01$
- MIRAGE-Ada: adaptive mirror descent, $\lambda=0.01$
- MIRAGE-IS: Itakura-Saito divergence, $\lambda=0.01$

### Evaluation Protocol

**5-fold cross-validation** with shuffling (`KFold`, `random_state=42`).

Metrics per model per dataset:
- `test_mse`: out-of-sample MSE (primary)
- `train_mse`: in-sample MSE (overfitting proxy)
- `test_r2`: coefficient of determination
- `weight_entropy`: $H(\theta)$
- `hhi`: Herfindahl-Hirschman index
- `enb`: effective number of bets
- `weight_cosine`: cosine similarity to ground-truth (where available)
- `weight_l1`: L1 distance from ground-truth
- `time_s`: wall-clock training time

### Running

```bash
python benchmark.py                   # full benchmark
python benchmark.py --dataset alpha   # single dataset
python benchmark.py --quick           # fewer iterations
python benchmark.py --folds 3 --seed 0
```

---

## 2. Ablation Studies (`ablation.py`)

Eight controlled ablations, each varying one component while holding others fixed.

### Study 1: Lambda Sensitivity

Sweeps $\lambda \in [10^{-4}, 1.0]$ (25 log-spaced values). Reports:
- CV MSE, weight entropy $H(\theta)$, HHI, ENB
- Identifies the optimal $\lambda$ and the MSE–diversity tradeoff curve

### Study 2: Optimiser Comparison

Compares all 4 optimisers on a single fixed dataset with matched hyperparameters. Reports MSE, training time, convergence speed, and final weight distribution.

### Study 3: Bregman Divergence Comparison

Compares KL, Euclidean, Itakura-Saito, and Beta($\beta=1.5$) divergences. Reports final MSE, entropy, and convergence behaviour.

### Study 4: Learning Rate Sensitivity

Sweeps $\eta \in [10^{-3}, 1.0]$ for each optimiser. Identifies the stable learning-rate range and divergence boundary.

### Study 5: Dimensional Scaling

Increases $n \in \{10, 20, 50, 100, 200, 500\}$ with fixed $m = 500$. Reports:
- Empirical MSE vs $n$
- Theoretical KL bound vs Euclidean bound
- KL/Euclidean ratio demonstrating the $\sqrt{n/\log n}$ advantage

### Study 6: Sample Scaling

Increases $m \in \{100, 500, 1000, 5000, 10000\}$ with fixed $n = 20$. Reports MSE improvement rate and training time scaling.

### Study 7: Noise Robustness

Varies SNR = $\{1, 5, 10, 20, 50, 100\}$. Compares MIRAGE++ vs OLS across the noise spectrum.

### Study 8: Correlation Robustness

Varies feature correlation $\rho \in \{0, 0.3, 0.6, 0.9, 0.95, 0.99\}$ using equicorrelation structure. Tests stability under near-collinearity.

### Running

```bash
python ablation.py                              # all 8 studies
python ablation.py --study lambda optimizer     # selected studies
python ablation.py --quick                      # faster run
```

---

## 3. Market Data Experiments (`examples/market_experiments.py`)

### Experiment 1: S&P 500 Sector ETF Portfolio

**Data:** 11 SPDR sector ETFs (XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLB, XLU, XLRE, XLC), daily log-returns 2015–2024.

**Task:** Given lagged sector returns (1-day lag, no look-ahead), predict SPY daily return. MIRAGE++ learns a diversified sector allocation satisfying the no-shorting, no-leverage constraint.

**Evaluation:** 5-fold time-series CV (no shuffling). Lambda sensitivity analysis.

### Experiment 2: Fama-French 5-Factor Combination

**Data:** FF5 factors (Mkt-RF, SMB, HML, RMW, CMA) and SPY monthly excess returns 2010–2024.

**Task:** Estimate simplex factor loadings — a budget-constrained factor exposure that sums to 1. The simplex constraint has economic meaning: it prevents negative factor exposures and ensures interpretable diversification across risk premia.

### Experiment 3: Technical Indicator Ensemble

**Data:** 12 technical signals computed from SPY OHLCV history:

| Signal | Formula |
|--------|---------|
| `mom_5`, `mom_21`, `mom_63`, `mom_126` | Price momentum (various windows) |
| `rsi_14` | 14-day Relative Strength Index |
| `ma_cross_20_50` | MA(20)/MA(50) ratio (trend) |
| `ma_cross_50_200` | MA(50)/MA(200) ratio (long-trend) |
| `vol_ratio_5_20` | Realised vol ratio (short/long) |
| `vol_ratio_10_60` | Realised vol ratio (wider) |
| `reversal_5` | Short-term mean reversion (-mom_5) |
| `vs_52w_high` | Distance from 52-week high |
| `vs_52w_low` | Distance from 52-week low |

All signals z-scored via 60-day rolling normalisation.

**Evaluation:** 5-fold time-series CV + rolling-window backtest (252-day window, 21-day refit). Tracks weight evolution and ENB over time.

### Experiment 4: Equity Universe (30 Stocks)

**Data:** 30 large-cap S&P 500 constituents, daily returns 2018–2024.

**Task:** Learn portfolio weights over 30 stocks that best predict SPY return. Higher-dimensional version of Experiment 1; tests scaling behaviour and concentration vs Lasso/Ridge.

---

## 4. Time-Series Cross-Validation

All market data experiments use **`TimeSeriesSplit`** (sklearn) which preserves temporal order:

```
Fold 1:  [=====train=====] [=test=]
Fold 2:  [======train======] [=test=]
Fold 3:  [=======train=======] [=test=]
...
```

This prevents data leakage: the model never sees future data during training. The 1-day feature lag further ensures no look-ahead bias.

---

## 5. Reproducibility

All synthetic experiments use fixed random seeds (`seed=42` by default). Market data is cached to `data/` after first download — running experiments without internet access is fully supported as long as the cache is populated.

Full environment: Python 3.10+, NumPy, SciPy, pandas, scikit-learn, matplotlib, yfinance, pandas-datareader. See `requirements.txt`.
