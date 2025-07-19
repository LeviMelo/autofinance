# Advanced Automated Portfolio Allocation System for the Brazilian (B3) Market

**Full Technical Project Specification & Implementation Guide**
**Status:** Design Blueprint (Proof-of-Concept → Production-Grade Path)
**Authoring Context:** Jupyter-friendly research stack; modular Python package; data pulled programmatically (BCB SGS, yfinance, CVM filings via aggregator scrapers).
**Timezone Assumption:** America/São\_Paulo (BRT/BRST-aware).
**Target Universe:** Brazilian equities (free-float adjusted common + preferred shares as needed; liquid tickers, e.g., PETR4.SA, VALE3.SA, ITUB4.SA, etc.).
**Benchmark:** IBOVESPA (^BVSP).
**Risk-Free:** SELIC (series 11 @ BCB/SGS).

---

## Table of Contents

1. [Project Goals & Philosophy](#project-goals--philosophy)
2. [Scope, Assumptions & Non-Goals](#scope-assumptions--non-goals)
3. [Conceptual Architecture Overview](#conceptual-architecture-overview)
4. [Data Governance & Acquisition Layer](#data-governance--acquisition-layer)

   * 4.1 Data Sources
   * 4.2 Data Frequency Harmonization
   * 4.3 Symbol Master & Corporate Action Hygiene
   * 4.4 Data Validation Rules
   * 4.5 ETL Pipelines & Storage Schema
5. [Return Construction & Preprocessing](#return-construction--preprocessing)

   * 5.1 Price-to-Return Conventions
   * 5.2 Excess Return Construction
   * 5.3 Handling Dividends, Splits, and Events
   * 5.4 Missing Data, Illiquidity & Outliers
6. [Factor Construction (B3-Specific Fama-French 3)](#factor-construction-b3-specific-fama-french-3)

   * 6.1 Universe Filters for Factor Portfolios
   * 6.2 Size Breakpoints → SMB
   * 6.3 Book-to-Market Breakpoints → HML
   * 6.4 Daily Factor Returns from Quarterly Fundamentals
   * 6.5 Quality Control & Sanity Checks
7. [Module 1: Dynamic Risk Engine (Σ Forecast)](#module-1-dynamic-risk-engine--σ-forecast)

   * 7.1 Statistical Foundations
   * 7.2 Workflow: GARCH → Standardized Residuals → DCC → Covariance Path
   * 7.3 Ledoit-Wolf Shrinkage Integration
   * 7.4 Rolling Estimation Mechanics
   * 7.5 Stress Overrides & Regime Conditioning (Optional Extension)
8. [Module 2: Return Engine (μ Views)](#module-2-return-engine-μ-views)

   * 8.1 View 1: Fama-French Beta × Historical Premia Projection
   * 8.2 View 2: VAR(p) Multivariate Statistical Forecast
   * 8.3 Optional View Extensions (Momentum, Macro, Machine Learning)
   * 8.4 View Harmonization (Annualization, Horizon Scaling, Timing Alignment)
9. [Module 3: Black-Litterman Synthesizer](#module-3-black-litterman-synthesizer)

   * 9.1 Mathematical Formulation (Canonical)
   * 9.2 Encoding Multi-Model Views (Factor & VAR)
   * 9.3 Building P and Q Matrices Programmatically
   * 9.4 View Confidence Calibration (Ω)
   * 9.5 Scaling τ & Interaction with Shrunk Σ
   * 9.6 Posterior μ\_BL Diagnostics
10. [Module 4: Portfolio Optimizer](#module-4-portfolio-optimizer)

    * 10.1 Optimization Problem Forms
    * 10.2 Constraints (Budget, Long-Only, Sector Caps, Liquidity Screens)
    * 10.3 Transaction Costs & Turnover Penalties
    * 10.4 Numerical Stability Strategies
    * 10.5 Solver API & Fallback Logic
11. [Backtesting Engine](#backtesting-engine)

    * 11.1 Rolling-Window Simulation Framework
    * 11.2 Data Windows & Rebalance Logic
    * 11.3 Portfolio Holding Period Return Realization
    * 11.4 Cash Handling, Slippage, Delistings
    * 11.5 Parallelization & Caching
12. [Performance Analytics & Attribution](#performance-analytics--attribution)

    * 12.1 Core Metrics
    * 12.2 Risk Decomposition
    * 12.3 Factor Attribution vs. IBOVESPA
    * 12.4 Timing vs. Selection Effects
    * 12.5 Bayesian Forecast Skill Diagnostics
13. [Visualization & Reporting Layer](#visualization--reporting-layer)

    * 13.1 Jupyter Markdown Report Blocks
    * 13.2 Interactive vs. Static Modes
    * 13.3 Dashboard Panels by Module
    * 13.4 Repeatable Notebook-to-HTML/PDF Publishing
14. [Project Repository Structure](#project-repository-structure)
15. [Configuration & Parameter Management](#configuration--parameter-management)
16. [Testing Strategy & Validation Milestones](#testing-strategy--validation-milestones)
17. [Computational Performance & Scaling](#computational-performance--scaling)
18. [Data Ethics, Compliance & Auditability](#data-ethics-compliance--auditability)
19. [Roadmap: Proof-of-Concept → Research Prototype → Production](#roadmap-proof-of-concept--research-prototype--production)
20. [Appendices](#appendices)

    * A. Mathematical Reference
    * B. API Notes (BCB SGS, yfinance, CVM scraping targets)
    * C. Data Schema DDL Draft
    * D. Example Config YAMLs
    * E. Glossary (Finance, Stats, Code)

---

## Project Goals & Philosophy

This system seeks to **separate signal generation from risk estimation**, **combine heterogeneous forecasts**, and **discipline subjective modeling with Bayesian blending**. The overall aim is not to beat every competing strategy but to **demonstrate a robust, reproducible, academically grounded, and practically auditable research stack** for Brazilian equity allocation that avoids common pitfalls:

* **No naive use of historical sample means** as forward μ.
* **No static covariance**; risk is time-varying.
* **Multi-model expected returns** → economic factor structure + statistical interdependence.
* **Bayesian fusion** (Black-Litterman) to temper noisy views with market equilibrium priors.
* **Shrinkage-stabilized covariance** to ensure invertibility and reduce overfitting.
* **Rolling out-of-sample evaluation** to validate real-world decision utility.
* **Transparent reporting**: every rebalance decision is reconstructible from logged inputs and parameters.

---

## Scope, Assumptions & Non-Goals

### In-Scope (PoC Phase)

* Daily adjusted total-return series for \~20–100 liquid B3 equities (parameterizable).
* IBOVESPA as market proxy; market-cap weights as w\_mkt (free-float if available; close alternative accepted at PoC).
* Daily SELIC interpolation to trading calendar.
* Fama-French 3 analog computed from B3 fundamentals (quarterly book value; latest shares outstanding).
* GARCH(1,1) + DCC correlation + Ledoit-Wolf shrink. (Initial: Student-t optional later.)
* VAR forecasts (order selected by AIC/BIC or rolling cross-validation).
* Black-Litterman posterior construction with programmatically scaled confidences.
* Markowitz allocation under long-only and fully invested constraints (baseline).
* Monthly rebalance rolling backtest with 5y lookback default.

### Extended (Research Phase)

* T-cost modeling (bps per turnover × illiquidity scaling).
* Shorting & leverage (subject to financing cost vs. SELIC).
* Regime-switching risk models (crisis detection).
* Additional factors (quality, momentum, profitability, carry from FX exposure of ADR pairs).
* Hierarchical risk parity comparison baseline.

### Out-of-Scope (Initial PoC)

* Intraday data / HFT signals.
* Derivatives (futures, options) exposures.
* Real-time production-grade failover.
* Tax-aware rebalancing (may be annotated but not modeled initially).

---

## Conceptual Architecture Overview

The system is a **layered research platform** built around reproducible data ingestion, modular modeling, and rigorous out-of-sample simulation.

**Layers:**

1. **Acquisition** → Raw data ingestion (prices, index, SELIC, fundamentals).
2. **Data Warehouse** → Clean, gap-aware, corporate-action-adjusted tables; factor-ready panels.
3. **Feature Engines**

   * Risk Engine (Σ forecast path).
   * Return Engines (μ views; Factor & VAR; pluggable new views).
4. **Bayesian Fusion (Black-Litterman)** → Posterior μ\_BL.
5. **Optimization Layer** → Solve for w\*.
6. **Execution Simulator** → Apply weights at rebalance; hold; realize PnL.
7. **Analytics & Reporting** → Metrics, attribution, visual diagnostics.

**Data flows are time-index-aware** to prevent look-ahead bias. **All intermediate artifacts are versioned** (Data version + Code version + Parameter hash).

---

## Data Governance & Acquisition Layer

### 4.1 Data Sources

| Data Type              | Frequency                  | Unit         | Primary Source                              | Backup                    | Notes                                                                                  |
| ---------------------- | -------------------------- | ------------ | ------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------- |
| Adjusted Equity Prices | Daily                      | BRL          | yfinance (Ticker.SA)                        | B3/Broker API             | Must include adj close; verify dividend reinvestment.                                  |
| IBOVESPA Index (^BVSP) | Daily                      | BRL          | yfinance                                    | B3 data direct            | Also fetch constituent history when possible.                                          |
| SELIC (risk-free)      | Daily (will downscale)     | % annualized | BCB SGS series 11                           | Banco Central CSV dumps   | Convert to daily risk-free return on B3 calendar.                                      |
| Shares Outstanding     | Quarterly (or event-based) | Shares       | CVM filings (DFP/ITR) via aggregator scrape | StatusInvest, Fundamentus | Free-float vs. total; choose consistent def.                                           |
| Book Value Equity      | Quarterly                  | BRL          | CVM filings                                 | Aggregators               | Use consolidated equity attributable to shareholders; convert to per-share book value. |

**Data Licensing Note:** Aggregators scrape public CVM filings; confirm ToS for re-use; for PoC restrict to academic/fair-use.

---

### 4.2 Data Frequency Harmonization

**Challenge:** Prices = Daily; Fundamentals = Quarterly; SELIC = effective annual / daily base.

**Approach:**

* Build a canonical **trading calendar** for B3 (holidays, early closes).
* Resample fundamentals to **daily forward-fill within quarter** (with event-date lag to avoid look-ahead: apply new financials only after official publication + settlement lag, e.g., 3 business days).
* Align SELIC to trading days by **day-count conversion**: $r_{f,daily} = (1 + r_{annual})^{1/252} - 1$ or use 252 trading days; alternatively use business-day compounding from CDI-style daily factor (preferred once CDI curve integrated).
* Ensure all time series indexed in **timezone-naïve UTC midnight** but logically mapped to São Paulo trading day; document transformation.

---

### 4.3 Symbol Master & Corporate Action Hygiene

Maintain a **symbol master table** with:

* `ticker_b3` (e.g., PETR4.SA)
* `isin`
* `cvm_code`
* `currency`
* `listing_start`, `listing_end`
* `corporate_action_history` (JSON: splits, symbol changes, mergers)
* `free_float_pct` (time series if available)
* `sector_classification` (B3 level 1 & 2; GICS mapping optional)

**Corporate Actions:**

* Validate yfinance adj factors against B3 event data.
* Build **split/dividend audit**: given raw close + events reconstruct total-return index; reconcile with provider `Adj Close`; flag discrepancies > tolerance (e.g., 5 bps cumulative drift).
* Use **survivorship-bias-free security master**: include delisted names historically; map to NA after delist date; zero weights beyond exit.

---

### 4.4 Data Validation Rules

Automated QA checks after ingestion:

* **Price monotonic sanity**: No <0; extreme >+/-80% day moves flagged vs. corporate action context.
* **Missing block detection**: >5 consecutive missing trading days triggers liquidity filter flag.
* **Currency stability**: Ensure all are BRL; convert if needed (rare cross-listing issues).
* **Fundamental plausibility**: Book equity ≥0 (flag negative); shares outstanding nonzero; step changes aligned to known corporate events.
* **Return outlier winsorization**: Pre-risk-model z-score clipping optional (document if used).

All checks logged; failing records quarantined; pipeline can run with partial coverage but will drop assets failing gating rules.

---

### 4.5 ETL Pipelines & Storage Schema

**Storage Options:**

* *PoC:* Parquet partitioned by data type/date.
* *Research:* DuckDB or SQLite analytic store.
* *Production:* PostgreSQL + columnar cache (DuckDB/Parquet).

**Canonical Tables (wide or long as needed):**

**`prices_equity_daily`** (long)

* date
* ticker
* adj\_close
* adj\_open (optional)
* adj\_high/low (optional)
* volume
* total\_return\_factor (cumulative; optional)

**`index_ibov_daily`**

* date
* adj\_close
* index\_divisor (if reconstructing)
* constituent\_weight\_json (snapshot when available)

**`risk_free_daily`**

* date
* selic\_annualized
* rf\_daily (compounded to trading day)

**`fundamentals_quarterly`**

* fiscal\_period\_end
* publish\_date
* ticker
* shares\_outstanding
* book\_equity
* book\_per\_share (calc)
* market\_cap\_at\_publish (join to price)

**`factor_panel_daily`** (output)

* date
* mkt\_excess
* smb
* hml

**`universe_membership_daily`**

* date
* ticker
* eligible\_flag
* float\_mcap
* liquidity\_decile

**Data Lineage Columns**: `source`, `ingest_ts`, `hash_raw`, `hash_clean`.

---

## Return Construction & Preprocessing

### 5.1 Price-to-Return Conventions

Let $P_t$ = adjusted close (split & dividend adjusted total-return price proxy).
Daily simple return: $r_t = \frac{P_t}{P_{t-1}} - 1$.
Log return: $\ell_t = \ln(P_t) - \ln(P_{t-1})$.
Use **log returns** for modeling (GARCH/VAR) but **compound simple returns** for performance simulation.

### 5.2 Excess Return Construction

Excess return vs. daily risk-free:
$r_{i,t}^{ex} = r_{i,t} - r_{f,t}$.
Use matched trading calendar; forward-fill rf across weekends/holidays.

### 5.3 Handling Dividends, Splits, and Events

If using provider adj\_close is reliable, accept; else reconstruct:
$P^{TR}_t = P^{PX}_t \times \prod_{events \le t}(1 + \frac{DIV_d}{P^{PX}_{d-1}}) / SplitFactor$.
Log audit diff vs. provider.

### 5.4 Missing Data, Illiquidity & Outliers

* **Stale Prices:** If no price change > X days, treat as zero return or drop asset? Default: mark missing; require min liquidity threshold in universe screen.
* **Short Gaps:** Forward fill up to 2 days; beyond → missing block; asset excluded at rebalance if insufficient history.
* **Minimum History Rule:** Require ≥ lookback\_window length (e.g., 5y) or fallback shrink strongly if partial.
* **Outliers:** Pre-model clipped at ±5σ of rolling 63d window; record pre- vs. post-clean.

---

## Factor Construction (B3-Specific Fama-French 3)

We implement **Brazil-local SMB & HML** factors; Market factor is IBOV excess vs. SELIC.

### 6.1 Universe Filters for Factor Portfolios

At each **factor formation date** (quarterly when new fundamentals publish; applied after publication lag):

* Eligible: Common + preferred with valid price series, >X% trading days, positive book eq (unless including distressed bucket).
* Rank by **float market cap** (shares\_outstanding × price).
* Rank by **book-to-market** = book\_equity / market\_cap (or book\_per\_share / price).

### 6.2 Size Breakpoints → SMB

* Compute size median across eligible names.
* Portfolio S = average return of “small” (below median).
* Portfolio B = average return of “big” (above median).
* SMB = S − B (value-weighted or equal-weight? choose config; default value-weight by float\_mcap).

### 6.3 Book-to-Market Breakpoints → HML

* Use 30/40/30 split or top/bottom tertiles.
* H = top 30% B/M (value).
* L = bottom 30% B/M (growth).
* HML = H − L (value-weighted by float\_mcap within buckets).

### 6.4 Daily Factor Returns from Quarterly Fundamentals

Procedure:

1. At fundamental publish + lag, recompute factor membership.
2. Freeze membership until next update.
3. Compute **daily portfolio returns** by compounding constituent asset daily returns with **rebalanced-to-weights at formation date** (no daily rebal unless using float market-cap drift—configurable).
4. Output daily SMB and HML series.
5. Market factor: $MKT = r_{IBOV} - r_f$.

### 6.5 Quality Control & Sanity Checks

* Does SMB show small-cap premium historically? Plot cumulative SMB.
* Correlations: SMB vs. HML low-to-moderate ideally; confirm not collinear.
* Cross-check vs. any published B3 factor library (if accessible; for research only).
* Rolling regressions: Are betas stable? Flag assets w/ unstable loadings.

---

## Module 1: Dynamic Risk Engine (Σ Forecast)

### 7.1 Statistical Foundations

We want a **time-varying conditional covariance matrix** for N assets: $\Sigma_t = D_t R_t D_t$, where:

* $D_t = \text{diag}(\sigma_{1,t}, ..., \sigma_{N,t})$
* $R_t$ = correlation matrix from DCC.

### 7.2 Workflow: GARCH → Standardized Residuals → DCC → Covariance Path

**Step 1: Fit univariate GARCH(1,1)** to each log excess return series:

$$
r_{i,t} = \mu_i + \epsilon_{i,t}, \quad \epsilon_{i,t} = \sigma_{i,t} z_{i,t}
$$

$$
\sigma_{i,t}^2 = \omega_i + \alpha_i \epsilon_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2
$$

* Distribution: Gaussian (baseline) → Student-t extension optional.
* Parameter estimation via MLE in rolling window; store last conditional variance forecast $\hat\sigma_{i,t+1}^2$.

**Step 2: Standardize residuals**:

$$
u_{i,t} = \frac{\epsilon_{i,t}}{\hat\sigma_{i,t}}
$$

**Step 3: Fit DCC(1,1)** to multivariate standardized residual matrix $U_t$:

$$
Q_t = (1 - a - b) \bar{Q} + a (u_{t-1} u_{t-1}^\top) + b Q_{t-1}
$$

$$
R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
$$

where $\bar{Q}$ is unconditional correlation of $U$ over the window.

**Step 4: Reconstruct conditional covariance**:

$$
H_t = D_t R_t D_t
$$

Use forecasted $D_{t+1}$ (from GARCH) and $R_{t+1}$ (from DCC recursion) to form **forward Σ forecast**.

### 7.3 Ledoit-Wolf Shrinkage Integration

Motivation: DCC-GARCH estimates can be noisy for large N relative to sample window length. Apply **targeted shrinkage**:

* Target $F_t$: structured matrix. Options:

  1. **Constant Correlation**: mean off-diagonal correlation × vol diag.
  2. **Single-Factor Market Model**: $\beta_i \beta_j \sigma_m^2$ + idiosyncratic diag.
  3. **Identity Scaled**: diag(vol\_i^2), no cross-corr (aggressive shrink).
* Use Ledoit-Wolf shrink intensity $\delta_t$ estimated from sample variability; apply:

$$
\Sigma_t^{(shr)} = \delta_t F_t + (1 - \delta_t) H_t
$$

Store both raw and shrunk versions; risk model consumers use shrunk by default.

### 7.4 Rolling Estimation Mechanics

* Rolling window length = lookback\_window (e.g., 5y ≈ 1260 trading days).
* At rebalance date $t$, fit GARCH & DCC using data up to $t$ only.
* Optional **exp-weighted** log-likelihood weighting to emphasize recent volatility regimes.
* Cache fitted parameters to avoid full refits at each month; warm-start next fit.

### 7.5 Stress Overrides & Regime Conditioning (Optional Extension)

* Crisis Regime Flag (volatility spike detection via realized vol > threshold).
* Override shrink intensity upward under crisis.
* Blend with **long-run covariance** floor to avoid false diversification.

---

## Module 2: Return Engine (μ Views)

### 8.1 View 1: Fama-French Beta × Historical Premia Projection

For each asset i:

1. Regress rolling excess returns $r_i^{ex}$ on factors $[MKT, SMB, HML]$:

   $$
   r_{i,t}^{ex} = \alpha_i + \beta_{i,M} MKT_t + \beta_{i,S} SMB_t + \beta_{i,H} HML_t + \epsilon_{i,t}
   $$
2. Estimation window = same lookback as risk? Configurable (e.g., 3y for more responsive betas; 5y for stability).
3. Store betas and regression diagnostics (R², t-stats, residual variance).
4. Estimate **forward factor premia**:

   * Long-horizon historical average (geometric or arithmetic; specify).
   * Or shrink toward zero (Bayesian prior).
   * Optionally condition on macro regimes (future extension).
5. Compute **expected excess return view**:

   $$
   \hat\mu^{FF}_i = \beta_{i,M} \hat{E}[MKT] + \beta_{i,S} \hat{E}[SMB] + \beta_{i,H} \hat{E}[HML] + \hat{\alpha}_i^*
   $$

   Where $\hat{\alpha}_i^*$ may be set to 0 (pure factor view) or include a shrunk historical alpha.

### 8.2 View 2: VAR(p) Multivariate Statistical Forecast

* Use vector of N asset **log excess returns**.
* Select order p via information criteria (AIC/BIC), max p cap (e.g., 5).
* Fit:

  $$
  R_t = c + \sum_{k=1}^p A_k R_{t-k} + e_t
  $$
* One-step-ahead forecast:

  $$
  \hat{R}_{t+1}^{VAR} = c + \sum_{k=1}^p A_k R_{t+1-k}
  $$
* Convert to **expected excess simple returns** for BL integration:

  * `exp_log` → simple approx: $\hat{r} \approx e^{\hat{\ell}} - 1$.
* VAR stability diagnostics: eigenvalues <1; record; fallback to shrinked AR(1) if unstable.

### 8.3 Optional View Extensions (roadmap)

* **Time-Series Momentum**: k-month lookback risk-adjusted signal → expected reversion or continuation.
* **Macro-Conditional Expected Returns**: regression of asset clusters vs. macro factors (commodity prices, FX, rate spreads).
* **Machine Learning Ensemble**: Gradient boosting over engineered features (vol regime, earnings surprises, oil price for PETR, etc.).
* Each new view is encoded into BL with its own uncertainty.

### 8.4 View Harmonization (Annualization, Horizon Scaling, Timing Alignment)

* Factor view implicitly long-horizon (annual premia). Scale to rebalance horizon:
  $\mu_{horizon} = (1 + \mu_{annual})^{h/252} - 1$.
* VAR is 1-step daily; convert to multi-day (rebalance hold) using compounding or scale expected daily drift × horizon.
* Decide whether BL is combining **annualized** numbers or **period-aligned** expected returns. Recommended: convert all inputs to **expected excess return over next rebalance period** (e.g., 21 trading days for monthly).

---

## Module 3: Black-Litterman Synthesizer

### 9.1 Mathematical Formulation (Canonical)

Given:

* $\Sigma$: prior covariance (use shrunk Σ\_t forecast).
* $w_{mkt}$: market cap weights (float-weighted B3 universe).
* $\lambda$: risk aversion (inverse of market Sharpe; configurable).
* Equilibrium prior:

  $$
  \pi = \lambda \Sigma w_{mkt}
  $$
* Views:

  $$
  P \mu = Q + \text{error}, \quad error \sim N(0, \Omega)
  $$

Posterior mean:

$$
\mu_{BL} = \pi + \tau \Sigma P^\top (P \tau \Sigma P^\top + \Omega)^{-1}(Q - P \pi)
$$

Posterior covariance:

$$
\Sigma_{BL} = \Sigma + \left[ \Sigma - \tau \Sigma P^\top (P \tau \Sigma P^\top + \Omega)^{-1} P \tau \Sigma \right]
$$

(Exact form depends on convention; double-check implementation; many libraries return posterior uncertainty differently.)

### 9.2 Encoding Multi-Model Views (Factor & VAR)

We have **two view sets**:

**Factor-Based Expected Returns (per-asset)**: "Asset i expected excess return = μ\_FF\_i." This is a **full absolute view**; P row = unit vector selecting asset i; Q entry = μ\_FF\_i. But modeling N absolute views leads to large Ω and computational burden. Alternatives:

* **Relative Spread Views**: E.g., asset i − market = μ\_FF\_i − π\_i.
* **Clustered Views**: Sector-level average of factor-implied returns.

**VAR Forecasts**: Provide a second forecast per asset. Combine w/ factor views at same P but with different uncertainty; or fuse factor & VAR into **blended Q\_i before BL**?
Recommended: treat **each model as separate observation** → stack P; enlarge Ω block-diagonally.

**Stacking Example (N assets, two models):**

* P\_FF = I\_N

* Q\_FF = μ\_FF

* Ω\_FF = diag(σ\_FF\_i²) (confidence inverse; proxied by regression residual var / sample error)

* P\_VAR = I\_N

* Q\_VAR = μ\_VAR

* Ω\_VAR = diag(σ\_VAR\_i²) (forecast error variance from VAR residuals + parameter uncertainty)

Stack:

$$
P = \begin{bmatrix} I_N \\ I_N \end{bmatrix}, \quad Q = \begin{bmatrix} \mu_{FF} \\ \mu_{VAR} \end{bmatrix}, \quad \Omega = \begin{bmatrix} \Omega_{FF} & 0 \\ 0 & \Omega_{VAR} \end{bmatrix}
$$

### 9.3 Building P and Q Matrices Programmatically

Utility function:

```python
def build_identity_views(mu_vec, var_vec):
    # mu_vec: (N,) view levels
    # var_vec: (N,) variances
    P = np.eye(len(mu_vec))
    Q = mu_vec.reshape(-1, 1)
    Omega = np.diag(var_vec)
    return P, Q, Omega
```

Then stack across models. Validate shapes: P(K×N), Q(K×1), Ω(K×K).

### 9.4 View Confidence Calibration (Ω)

Crucial: correct **relative** scaling across view sets.
Candidate heuristics:

1. **Model RMSE-Based**: Use rolling out-of-sample error of each model; larger RMSE = larger Ω (lower confidence).
2. **t-Statistic Scaling**: Var ∝ 1 / t² of regression coefficients (factor model).
3. **Signal-to-Noise Ratio**: Estimate predictive R² at horizon; scale Ω inversely.
4. **User Override**: Config weight α\_model; set Ω\_model = base / α\_model.

Document the chosen rule; record values each rebalance.

### 9.5 Scaling τ & Interaction with Shrunk Σ

τ rescales uncertainty in the prior equilibrium. Common small value: 0.025–0.05.
Because we already shrink Σ, τ interacts: If Σ is conservative (shrunk heavily), you may set τ higher to allow views impact; else smaller τ to trust market.

Parameter sweep recommended in research; log realized posterior weight on views to calibrate.

### 9.6 Posterior μ\_BL Diagnostics

After computing μ\_BL:

* Compare vs. π (prior) and raw μ\_FF / μ\_VAR.
* Plot bar chart: prior vs. posterior shift.
* Compute **Kullback-Leibler divergence** between implied optimal portfolios under prior vs. posterior (risk aversion fixed).
* Track **Effective View Weight**: fraction of posterior deviation explained by each view block (decompose via linear algebra).

---

## Module 4: Portfolio Optimizer

### 10.1 Optimization Problem Forms

Baseline: **Max Sharpe (unconstrained)** →

$$
\max_w \frac{w^\top \mu_{BL}}{\sqrt{w^\top \Sigma w}}, \quad \text{s.t. } \sum_i w_i = 1, \ w_i \ge 0.
$$

Equivalent quadratic:

$$
\max_w \left( w^\top \mu_{BL} - \frac{\gamma}{2} w^\top \Sigma w \right)
$$

for some risk aversion γ (solve param sweep; link γ to target vol).

Alternate forms:

* **Target Return**: minimize variance s.t. $w^\top \mu_{BL} = \mu^*$.
* **Target Volatility**: maximize return s.t. $w^\top \Sigma w = \sigma^{*2}$.
* **Min Variance**: ignore μ; baseline risk control.

### 10.2 Constraints

* **Budget:** $\sum w_i = 1$.
* **Long-only:** $w_i \ge 0$.
* **Soft Sector Caps:** $w_{sector} \le c_{sector}$.
* **Single Name Cap:** $w_i \le c_{name}$ (e.g., 10%).
* **Liquidity Filter:** Exclude bottom X% average daily traded volume; or scale max weight by liquidity.
* **Tracking Error Constraint (vs. IBOV):** $(w-w_{IBOV})^\top \Sigma (w-w_{IBOV}) \le \sigma_{TE}^2$.

### 10.3 Transaction Costs & Turnover Penalties

Let $\Delta w = w_{new} - w_{old}$.
Add penalty: $C^\top |\Delta w|$ or quadratic $\Delta w^\top \Gamma \Delta w$.
Or incorporate **net rebalancing rule**: trade only if drift > threshold.

PoC: log turnover; evaluate slippage ex-post.

### 10.4 Numerical Stability Strategies

* Use **shrunk Σ**; ensure PSD via eigenclip (replace negative eigenvalues with ε).
* Add jitter λI if condition number high.
* Use **cvxpy** or **quadprog** wrappers; fallback to iterative projection gradient.

### 10.5 Solver API & Fallback Logic

* Primary: cvxpy ECOS or OSQP.
* Secondary: scipy.optimize minimize.
* Fallback: heuristic scaling from unconstrained tangency portfolio clipped to constraints.

All optimizer calls logged with status, iteration, KKT residual.

---

## Backtesting Engine

### 11.1 Rolling-Window Simulation Framework

We simulate **how the strategy would have been implemented at each rebalance date** without look-ahead.

**Inputs:** `start_date_backtest`, `end_date_backtest`, `lookback_years`, `rebalance_freq ('M')`, `universe_rule`, `model_params`.

### 11.2 Data Windows & Rebalance Logic

At each rebalance date $t$:

1. Determine **lookback start** = $t - L$ (L in trading days).
2. Extract history slice for all assets with sufficient data coverage (≥ x% non-missing).
3. Apply **universe screen** at t: liquidity, availability, fundamentals.
4. Build returns panel & factor panel using only info available up to t (apply publication lags).
5. Run **Module 1→4** to produce new weights $w_t$.

### 11.3 Portfolio Holding Period Return Realization

* Hold weights $w_t$ from t (close) to t+Δ (next rebalance).
* Realized simple return:

  $$
  R_{p,t \to t+\Delta} = \sum_i w_{i,t} \cdot R_{i,t\to t+\Delta}
  $$
* Re-normalize for **cash drag** if uninvested portion >0.
* Apply **transaction cost** at rebalance: $-\sum_i cost_i |\Delta w_i|$.

Store time series of:

* Pre-cost return
* Post-cost return
* Turnover
* Portfolio vol (realized)
* Universe breadth (N active).

### 11.4 Cash Handling, Slippage, Delistings

* If asset delists mid-hold: assume payout at last available price; redistribute to cash; cash earns rf until next rebalance.
* If trading halt: freeze price; path dependent; record unrealized risk.

### 11.5 Parallelization & Caching

* Many rolling fits; cache intermediate model fits keyed by `(asset, window_end)` hash.
* Dask / joblib parallel per rebalance date (outer loop).
* Persist factor/regression stats to disk to avoid refit.

---

## Performance Analytics & Attribution

### 12.1 Core Metrics

Computed over full and sub-periods (bull, bear, crisis segments):

* Cumulative return
* Annualized return (CAGR)
* Annualized vol
* Sharpe (excess vs. SELIC)
* Sortino (downside semidev)
* Max drawdown & Calmar
* Skew, kurtosis (fat-tail awareness)
* Rolling 12m Sharpe chart

### 12.2 Risk Decomposition

* Contribution to variance by asset: $w_i \sigma_i \rho_{i,p} / \sigma_p$.
* Marginal contribution to risk; component contribution.
* Tracking error vs. IBOV.

### 12.3 Factor Attribution vs. IBOVESPA

Regress portfolio excess returns vs. B3 Fama-French factors:

$$
R_{p}^{ex} = \alpha_p + \beta_M MKT + \beta_S SMB + \beta_H HML + \epsilon
$$

Compute annualized alpha & t-test; rolling 36m factor loadings.

### 12.4 Timing vs. Selection Effects

Brinson-like decomposition relative to IBOV:

* Allocation effect (sector overweight vs. sector return).
* Selection effect (intra-sector stock picking).
* Interaction residual.

### 12.5 Bayesian Forecast Skill Diagnostics

Back-test each **view model’s raw forecasts**:

* Correlation between forecast μ and subsequent realized return.
* Ranked IC (information coefficient).
* Calibration curves: bucket expected return deciles vs. realized.
* BL shrink impact: forecast dispersion pre vs. post.

---

## Visualization & Reporting Layer

### 13.1 Jupyter Markdown Report Blocks

Each backtest run autogenerates a structured markdown report section summarizing:

* Run metadata (dates, params, N assets).
* Model stability diagnostics.
* Key performance metrics table.
* Top overweight/underweight vs. IBOV over time.
* Regime plots (vol spikes, correlation crises).

### 13.2 Interactive vs. Static Modes

* **Interactive (development):** Plotly or Bokeh; hover for asset details; toggle factors.
* **Static (report archive / .md export):** Matplotlib PNG/SVG saved to `/reports/figures/YYMMDD_runid/`.
* Provide script to **convert notebook → Markdown + assets** reproducibly.

### 13.3 Dashboard Panels by Module

**Module 1 Diagnostics:**

* Rolling vol forecasts vs. realized vol (per asset).
* Average correlation heatmap at each rebalance; animated sparkline.
* Shrink intensity δ\_t over time.

**Module 2 Diagnostics:**

* Factor betas with CI bands.
* VAR residual autocorrelation; stability roots.
* Side-by-side μ\_FF vs. μ\_VAR scatter (color = sector; size = forecast confidence).

**Module 3 Diagnostics:**

* Prior π vs. Posterior μ\_BL bars.
* Contribution by view (waterfall).
* View uncertainty heatmap (Ω diag).

**Module 4 Diagnostics:**

* Portfolio weights stacked area across time.
* Active weights vs. IBOV.
* Turnover bars per rebalance; cumulative transaction cost.

**Performance:**

* Equity curve vs. IBOV (gross & net).
* Drawdown plot.
* Rolling Sharpe & alpha (36m window).
* Risk contribution bar chart.

### 13.4 Repeatable Notebook-to-HTML/PDF Publishing

* `nbconvert` pipeline with run metadata injection.
* Inline summary tables auto-rendered from pandas to Markdown.
* At run completion: produce `run_<timestamp>.md` + zipped figs archive.

---

## Project Repository Structure

A reproducible research codebase should be **package-structured** yet **notebook-friendly**.

```
b3_alloc_system/
├─ pyproject.toml
├─ README.md
├─ config/
│   ├─ base.yaml
│   ├─ universe_small.yaml
│   ├─ backtest_monthly_5y.yaml
├─ data/
│   ├─ raw/
│   │   ├─ prices/
│   │   ├─ fundamentals/
│   │   ├─ selic/
│   ├─ interim/
│   ├─ processed/
│   └─ metadata/
├─ notebooks/
│   ├─ 00_data_audit.ipynb
│   ├─ 01_factor_construction.ipynb
│   ├─ 02_risk_engine_validation.ipynb
│   ├─ 03_return_engine_validation.ipynb
│   ├─ 04_black_litterman_sandbox.ipynb
│   ├─ 05_optimizer_tests.ipynb
│   ├─ 06_full_backtest_driver.ipynb
│   └─ 07_results_report_template.ipynb
├─ src/
│   ├─ b3alloc/
│   │   ├─ __init__.py
│   │   ├─ config.py
│   │   ├─ utils_dates.py
│   │   ├─ data/
│   │   │   ├─ ingest_prices.py
│   │   │   ├─ ingest_selic.py
│   │   │   ├─ ingest_fundamentals.py
│   │   │   ├─ corporate_actions.py
│   │   │   └─ warehouse.py
│   │   ├─ preprocess/
│   │   │   ├─ returns.py
│   │   │   ├─ align.py
│   │   │   └─ clean.py
│   │   ├─ factors/
│   │   │   ├─ fama_french_b3.py
│   │   │   └─ factor_qc.py
│   │   ├─ risk/
│   │   │   ├─ garch.py
│   │   │   ├─ dcc.py
│   │   │   ├─ shrinkage.py
│   │   │   └─ risk_engine.py
│   │   ├─ returns/
│   │   │   ├─ ff_view.py
│   │   │   ├─ var_view.py
│   │   │   └─ view_utils.py
│   │   ├─ bl/
│   │   │   ├─ black_litterman.py
│   │   │   ├─ view_builder.py
│   │   │   └─ confidence.py
│   │   ├─ optimize/
│   │   │   ├─ mean_variance.py
│   │   │   ├─ constraints.py
│   │   │   ├─ costs.py
│   │   │   └─ solver.py
│   │   ├─ backtest/
│   │   │   ├─ engine.py
│   │   │   ├─ portfolio_accounting.py
│   │   │   ├─ transaction_costs.py
│   │   │   └─ analytics.py
│   │   ├─ viz/
│   │   │   ├─ plots_risk.py
│   │   │   ├─ plots_views.py
│   │   │   ├─ plots_portfolio.py
│   │   │   └─ report_builder.py
│   │   └─ tests/
│   │       ├─ test_data_ingest.py
│   │       ├─ test_factor_construction.py
│   │       ├─ test_risk_engine.py
│   │       ├─ test_bl.py
│   │       └─ test_backtest.py
├─ reports/
│   ├─ runs/
│   │   └─ run_YYYYMMDD_HHMM/
│   │       ├─ report.md
│   │       ├─ figs/
│   │       └─ artifacts/
└─ scripts/
    ├─ run_backtest.py
    ├─ build_factors.py
    ├─ update_data.py
    └─ generate_report.py
```

---

## Configuration & Parameter Management

Centralize all tunables in YAML; load via pydantic schema for validation.

**Example high-level structure:**

```yaml
data:
  start: 2010-01-01
  end: 2025-01-01
  tickers_file: config/universe_small.csv
  selic_series: 11
  publish_lag_days: 3

risk_engine:
  garch:
    dist: gaussian
    min_obs: 500
    refit_freq_days: 21
  dcc:
    a_init: 0.02
    b_init: 0.97
  shrinkage:
    method: ledoit_wolf_constant_corr
    floor: 0.05

return_engine:
  factor:
    lookback_days: 756
    include_alpha: false
    premium_estimator: long_term_mean
  var:
    max_lag: 5
    criterion: bic
    log_returns: true

black_litterman:
  tau: 0.05
  confidence:
    factor_scaler: 1.0
    var_scaler: 0.5
    method: rmse_based

optimizer:
  objective: max_sharpe
  long_only: true
  name_cap: 0.10
  sector_cap: 0.25
  turnover_penalty_bps: 5

backtest:
  lookback_years: 5
  rebalance: monthly
  start: 2012-01-01
  end: 2025-01-01
  costs_bps: 10
```

---

## Testing Strategy & Validation Milestones

### Unit-Level

* **Data Ingest:** Known CSV fixture → expected DataFrame shape; corporate action adjustment test.
* **Returns:** Edge case forward-fill; missing price; log vs. simple parity test.
* **Factor Construction:** Controlled synthetic fundamentals; expected SMB sign.
* **Risk Engine:** Compare single-asset case vs. baseline GARCH library; PSD test on Σ.
* **BL Math:** Synthetic 2-asset example w/ closed-form expected posterior.

### Integration-Level

* Run mini-universe (3 assets, 1y) end-to-end; inspect logs.
* Introduce synthetic known alpha → confirm optimizer overweight.

### Backtest Validation

* **No-View Mode:** Set Ω → ∞ (ignore views). Confirm weights = market prior under BL; portfolio tracks IBOV.
* **Perfect-Foresight Sanity:** Replace μ views w/ realized next-month returns → confirm large performance (upper bound).
* **Stress Test:** Simulate crisis (2015 recession, 2020 COVID crash) → correlation spike detection.

---

## Computational Performance & Scaling

### Data Size Estimates

* 100 tickers × 15y daily ≈ <10 MB numeric; light.
* GARCH fit per asset monthly; acceptable CPU.
* DCC on 100-dim may be heavy; consider:

  * Reduced universe for PoC (N=25).
  * Factor-analytic DCC (project residuals to PCs).
  * Block-diagonal DCC by sector.

### Parallelization

* Thread/Process per asset for GARCH.
* Use vectorized DCC from `arch` or `mgarch` libs; if too slow, approximate with **EWMA correlation** fallback.

### Caching

* Serialize fitted model params; warm-start at next rebalance.
* Persist Σ forecast path to disk; skip recomputation in analysis notebooks.

---

## Data Ethics, Compliance & Auditability

* Respect CVM/regulatory usage; derivative publication of extracted data limited to internal research.
* Full **data lineage logs**: source timestamp, hash of raw file, transform version.
* Reproducibility: storing config snapshot + git commit hash with each backtest run.
* Provide **rebuild script** that reconstructs any portfolio weight vector from raw inputs & config.

---

## Roadmap: Proof-of-Concept → Research Prototype → Production

### Phase 0 – Foundations (Data & Utilities)

**Deliverables:** Trading calendar; price ingest; SELIC ingest; fundamentals ingest (min 10 tickers).
**Validation:** Price audit vs. corporate actions; rf alignment.

### Phase 1 – Factors

Build B3 Fama-French factors; sanity plots; regression of PETR4 vs. factors; compare to literature.

### Phase 2 – Risk Engine MVP

Implement univariate GARCH only; compare vs. rolling sample vol; upgrade to DCC; integrate shrinkage; PSD tests.

### Phase 3 – Return Views

Implement Factor μ; implement VAR μ; create view diagnostics; historical forecast error store.

### Phase 4 – Black-Litterman Integration

Stack views; calibrate Ω; run small static test; inspect posterior shifts.

### Phase 5 – Optimizer

Max Sharpe long-only; add weight caps; add turnover penalty; unit tests.

### Phase 6 – Rolling Backtest

Monthly rebalance; 5y lookback; generate performance metrics; compare to IBOV.

### Phase 7 – Visualization & Reporting Automation

Automated Markdown/HTML reports per run; plots; parameter summary.

### Phase 8 – Extensions

Costs, regime models, alternative priors, shorting, multi-asset (FX, rates).

---

## Appendices

---

### Appendix A. Mathematical Reference

**Annualization Conventions**
Trading days per year: use 252; document if 250/253 used.

**Converting Annual SELIC to Daily Rate**
If SELIC provided as annual effective rate $r_a$:

$$
r_{daily} = (1 + r_a)^{1/252} - 1
$$

**Reverse Optimization for Equilibrium Returns**
Given market cap weights $w_m$, risk aversion λ, and covariance Σ:

$$
\pi = \lambda \Sigma w_m
$$

Solve λ from historical market Sharpe:

$$
\lambda = \frac{E[R_m - R_f]}{Var(R_m - R_f)}
$$

Or user override.

**Ledoit-Wolf Shrinkage Coefficient** (sketch)
Let $S$ sample covariance; $F$ target; shrink intensity:

$$
\delta^\star = \frac{ \sum Var(s_{ij}) }{ \sum (s_{ij} - f_{ij})^2 }
$$

Clip 0–1.

**Black-Litterman Posterior** (alternative compact matrix form)

$$
\Sigma_{post} = \left[ (\tau \Sigma)^{-1} + P^\top \Omega^{-1} P \right]^{-1}
$$

$$
\mu_{post} = \Sigma_{post} \left[ (\tau \Sigma)^{-1}\pi + P^\top \Omega^{-1} Q \right]
$$

---

### Appendix B. API Notes (BCB SGS, yfinance, CVM Scraping)

**BCB SGS (Python snippet):**

* Endpoint: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series}/dados?formato=json`
* Series 11 = SELIC meta over? Confirm: sometimes SELIC meta vs. effective (Selic Over rate \~ series 4189). Validate which is appropriate for risk-free proxy; CDI may be more tradable; include config toggle.

**yfinance**

* Tickers must end `.SA`.
* Use `actions=True` to fetch dividends/splits.
* Rate limits; implement polite caching.

**CVM Filings**

* Official portal provides DFP/ITR XBRL; scraping heavier. Faster to parse aggregator (StatusInvest) HTML tables at PoC; mark as non-authoritative.

---

### Appendix C. Data Schema DDL Draft (SQLite Example)

```sql
CREATE TABLE prices_equity_daily (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    adj_close REAL,
    volume REAL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE risk_free_daily (
    date TEXT PRIMARY KEY,
    selic_annual REAL,
    rf_daily REAL
);

CREATE TABLE fundamentals_quarterly (
    fiscal_period_end TEXT,
    publish_date TEXT,
    ticker TEXT,
    shares_outstanding REAL,
    book_equity REAL,
    book_per_share REAL,
    PRIMARY KEY (ticker, fiscal_period_end)
);

CREATE TABLE factor_panel_daily (
    date TEXT PRIMARY KEY,
    mkt_excess REAL,
    smb REAL,
    hml REAL
);
```

---

### Appendix D. Example Parameter-Sweep Configs (Grid YAML Fragments)

**BL τ Sweep**

```yaml
experiments:
  - name: tau_005
    black_litterman: { tau: 0.05 }
  - name: tau_010
    black_litterman: { tau: 0.10 }
  - name: tau_020
    black_litterman: { tau: 0.20 }
```

**View Confidence Sweep**

```yaml
experiments:
  - name: factor_high_conf
    black_litterman:
      confidence:
        factor_scaler: 0.25
        var_scaler: 2.0
  - name: var_high_conf
    black_litterman:
      confidence:
        factor_scaler: 2.0
        var_scaler: 0.25
```

---

### Appendix E. Glossary (Finance, Stats, Code)

**Adj Close:** Price adjusted for splits + dividends (total-return synthetic).
**B3:** Brasil Bolsa Balcão; Brazil’s main stock exchange.
**Book-to-Market:** Book equity / market cap; value metric.
**DCC:** Dynamic Conditional Correlation; multivariate GARCH correlation process.
**GARCH:** Generalized Autoregressive Conditional Heteroskedasticity; time-varying volatility.
**Ledoit-Wolf Shrinkage:** Statistical method to improve covariance conditioning by shrinking sample matrix toward structured target.
**Black-Litterman:** Bayesian framework combining market equilibrium returns with subjective/model views.
**Sharpe Ratio:** Excess return / volatility.
**Sortino Ratio:** Excess return / downside deviation.
**Max Drawdown:** Largest peak-to-trough loss.
**Turnover:** % of portfolio traded at rebalance.
**Ω (Omega in BL):** View uncertainty covariance.
**τ (Tau in BL):** Scalar controlling relative weight of prior vs. views.

---

## Development Sequencing With Checkpoints

### Milestone 1 (Data Baseline: Week 1–2)

* Implement trading calendar.
* Download 10 tickers + IBOV + SELIC.
* Build return series; confirm no look-ahead.
* Output CSV audit.

**Deliverable:** `00_data_audit.ipynb`, QA logs.

---

### Milestone 2 (Fundamentals + Factors: Week 3–4)

* Scrape shares + book values for tickers across 5y.
* Construct quarterly panels; apply publication lag.
* Build SMB/HML; produce cumulative plots.
* Regression test: asset betas sensible.

**Deliverable:** `01_factor_construction.ipynb` + factor panel Parquet.

---

### Milestone 3 (Risk Engine MVP: Week 5–6)

* GARCH per asset; vol forecast vs. realized.
* DCC correlation path; heatmaps for 2015, 2020 crisis.
* Ledoit-Wolf shrink wrapper; PSD checks.

**Deliverable:** `02_risk_engine_validation.ipynb`.

---

### Milestone 4 (Return Views: Week 7–8)

* Factor-view μ; param toggles.
* VAR-view μ; horizon scaling.
* Compare view dispersion.

**Deliverable:** `03_return_engine_validation.ipynb`.

---

### Milestone 5 (Black-Litterman Integration: Week 9)

* Build P/Q from both views; calibrate Ω.
* Inspect posterior adjustments vs. prior.

**Deliverable:** `04_black_litterman_sandbox.ipynb`.

---

### Milestone 6 (Optimizer & Constraints: Week 10)

* Max Sharpe long-only; test weight caps.
* Stress Σ ill-conditioning; ensure solver robust.

**Deliverable:** `05_optimizer_tests.ipynb`.

---

### Milestone 7 (Full Rolling Backtest: Week 11–12)

* Monthly rebalance; 5y lookback; 2012–2025.
* Log performance; compare IBOV; produce metrics.

**Deliverable:** `06_full_backtest_driver.ipynb`.

---

### Milestone 8 (Report Automation & Extensions: Week 13+)

* Auto Markdown report w/ figs + metrics.
* Add turnover + cost sensitivity runs.
* Parameter sweeps (τ, Ω\_scalers).

**Deliverable:** `07_results_report_template.ipynb` → `/reports/runs/`.

---

## Implementation Details by Code Component

Below are more granular specifications to guide coding. These should map directly to module functions in `/src/b3alloc/`.

---

### Data Ingestion Functions

**`ingest_prices.fetch_yf_prices(tickers, start, end)`**

* Uses yfinance; downloads OHLCV+Adj Close.
* Reindexes to trading calendar; forward-fill missing corporate action days; mark missingness.
* Returns tidy DataFrame long-format.

**`ingest_selic.fetch_selic(series_id=11, start, end)`**

* Requests JSON from BCB SGS; parse date; convert to numeric.
* Interpolate to trading calendar; compute daily rf.

**`ingest_fundamentals.fetch_statusinvest_fundamentals(ticker)`**

* Scrape HTML; parse shares, book\_equity, fiscal period.
* Normalize to quarter end; record publish date if available; else infer typical lag.

---

### Preprocessing Utilities

**`returns.compute_log_returns(df_prices)`**
Returns panel pivoted (date × ticker) of log returns.

**`align.apply_publication_lag(fund_df, lag_days=3, calendar)`**
Shift factor membership start-date forward by lag.

**`clean.filter_liquid_universe(prices, min_trading_ratio=0.9, min_volume=None)`**
Return list of eligible tickers per date.

---

### Factor Engine

**`fama_french_b3.build_factor_memberships(fundamentals, prices, date)`**
Returns dict of {small,big,highBM,lowBM} membership arrays.

**`fama_french_b3.compute_daily_factor_returns(memberships, returns_panel)`**
Aggregates per bucket; constructs SMB/HML.

**`factor_qc.plot_factor_cumrets(factor_panel)`**
QC chart.

---

### Risk Engine

**`garch.fit_garch_series(r_series, dist='gaussian')`** → params, conditional σ² path.
**`dcc.fit_dcc(u_matrix)`** → a,b,Q\_t path.
**`shrinkage.ledoit_wolf(H)`** → δ, Σ\_shr.
**`risk_engine.build_covariance(date, returns_hist)`** → orchestrates above; returns Σ\_t\_shr + diagnostics.

---

### Return Views

**`ff_view.estimate_betas(returns_ex, factors_ex, window)`** → betas, residual var.
**`ff_view.forecast_mu(betas, factor_premia)`** → μ\_FF.
**`var_view.fit_var(R_matrix, max_lag, criterion)`** → model; forecast μ\_VAR; forecast variance diag.

---

### Black-Litterman

**`view_builder.stack_views(view_dicts)`** where each dict has `{name, mu, var}`.
**`black_litterman.posterior(mu_prior, Sigma, views, tau)`** → μ\_BL, Σ\_post.
**`confidence.estimate_view_uncertainty(residual_var, scaler)`**.

---

### Optimizer

**`constraints.build_constraint_mats(config, tickers, sector_map)`**.
**`mean_variance.max_sharpe(mu, Sigma, constraints)`**.
**`costs.apply_turnover_penalty(w_prev, w_new, bps)`** (or embed in objective).
**`solver.solve_qp()`** unified interface.

---

### Backtest Engine

**`engine.run_backtest(config)`** high-level driver:

1. Build rebalance dates.
2. For each date:

   * Slice hist.
   * Risk Engine → Σ.
   * Return Views → μ\_FF, μ\_VAR.
   * BL → μ\_BL.
   * Optimizer → w\_t.
   * Realize PnL to next date.
3. Aggregate results.

**`portfolio_accounting.roll_forward()`** to compute realized returns incl. costs.

**`analytics.compute_metrics(ret_series, rf_series, benchmark_series)`**.

---

### Visualization

**`plots_risk.plot_corr_heatmap(Sigma)`**.
**`plots_views.scatter_views(mu_FF, mu_VAR, mu_BL)`**.
**`plots_portfolio.weights_over_time(weights_df)`**.
**`report_builder.generate_markdown(run_artifacts)`** compile final doc.

---

## Data Integrity & Backtest Defensiveness Checklist

Before every backtest run, automatically assert:

| Check                          | Abort?                                 | Mitigation                    |
| ------------------------------ | -------------------------------------- | ----------------------------- |
| Missing SELIC in window        | Yes                                    | Fill via last obs; warn.      |
| \<N\_min assets meet liquidity | Config                                 | Drop run or continue flagged. |
| Σ not PSD                      | Auto-fix via eigenclip                 | Log severity.                 |
| Optimizer fail                 | Fallback to risk-parity or mkt weights | Record fallback flag.         |
| View NaNs                      | Replace with prior π\_i                | Log.                          |

All events recorded to `run_log.json`.

---

## Rebalancing Mechanics Clarification

**Timing Convention:**

* Use **close-to-close**: At rebalance date t, all signals use data through close t. Trades executed at next open (approximated using close t+1 if open not available); hold until next rebalance date t+Δ. For daily data PoC, we treat trade at t+1 close after weights computed end-of-day t (approximation; consistent across models). Document clearly.

**Alternative (Cleaner PoC):** Assume frictionless same-day execution at close t using info up to t-1. This avoids look-ahead in practice; adopt this standard unless realistic slippage modeling required.

---

## Handling of Survivorship & Delistings

* Build **historical membership** table from B3 delisting records (if available) or infer from price history.
* Do **not restrict to current survivors**; include names that delisted post-2015 etc.
* When delisted mid-hold: convert to cash; record realized return = final cash-out.

---

## Parameter Defaults & Justification

| Parameter                | Default         | Rationale                                                           | Tunable Range                   |
| ------------------------ | --------------- | ------------------------------------------------------------------- | ------------------------------- |
| Lookback Window          | 5y (\~1260d)    | Balance stability & responsiveness; Brazilian cycles \~ multi-year. | 2–10y                           |
| Rebalance                | Monthly         | Align w/ fundamental update cycles; manageable turnover.            | Weekly–Quarterly                |
| τ (BL)                   | 0.05            | Conventional small; tuned to shrink prior.                          | 0.01–0.25                       |
| Factor Premium Estimator | 10y mean        | Smooth cyclical noise; can shrink.                                  | 3y–full sample; Bayesian shrink |
| Ω scaling factor (FF)    | 1× residual var | Model-driven precision.                                             | 0.25–4×                         |
| Ω scaling factor (VAR)   | 2× residual var | Lower confidence in pure statistical view.                          | 1–10×                           |
| Name Cap                 | 10%             | Concentration control; fits B3 large-caps.                          | 5–25%                           |
| Sector Cap               | 25%             | Avoid energy/materials dominance.                                   | 15–35%                          |

---

## Risk Model Stress Experiments

To validate robustness:

1. **Vol Shock Injection:** Multiply last 20d returns by 2×; re-fit; measure Σ sensitivity.
2. **Corr Collapse:** Replace R\_t with 0.9 constant corr; see optimizer reaction.
3. **Shrinkage Off:** Compare performance Σ\_raw vs. Σ\_shr; track out-of-sample Sharpe stability.

---

## Forecast Skill Benchmarks

Because forecasting expected returns is notoriously noisy, track **model meta-metrics**:

| Metric                | Definition                                                   | Interpretation              |
| --------------------- | ------------------------------------------------------------ | --------------------------- |
| Forecast IC           | Spearman corr(μ\_view, realized next-period returns)         | >0 indicates ranking skill. |
| Hit Rate              | % assets correct sign vs. realized                           | Noise filter.               |
| Top-Decile Spread     | Mean realized of top decile − bottom decile by forecast rank | Economic separation.        |
| View Dispersion Ratio | Std(μ\_view) / Std(π)                                        | How aggressive vs. market.  |
| BL Dampening Ratio    | Std(μ\_BL − π) / Std(μ\_view − π)                            | Shrink magnitude.           |

---

## Logging & Metadata Standards

Every backtest run writes a JSON metadata record:

```json
{
  "run_id": "2025-07-17T1830Z",
  "git_commit": "abc1234",
  "config_hash": "f9d0...",
  "data_range": ["2010-01-01", "2025-01-01"],
  "num_assets_initial": 65,
  "num_assets_active_mean": 48,
  "rebalance_freq": "M",
  "tau": 0.05,
  "omega_scalers": {"factor":1.0,"var":2.0},
  "costs_bps": 10,
  "notes": "baseline PoC"
}
```

This file plus the config YAML guarantee reproducibility.

---

## Security & Reliability Considerations (Production Path)

* API keys (if broker feed) stored in `.env` + secrets manager.
* Daily cron data updates; QA emails on anomalies.
* Snapshot backup of processed data before each model run.
* Re-run detection: identical config/data hash → avoid duplication.

---

## Practical Development Tips

* Start with **very small universe** (5 tickers) to debug pipelines.
* Persist intermediate artifacts (Σ\_t, μ\_FF, μ\_VAR) to disk; build BL offline.
* Plot every intermediate series the first time; trust nothing unvisualized.
* Add **assert sorted index** checks everywhere; time alignment bugs ruin forecasts.
* Track **units (annual vs. daily)** in column names (suffix `_ann`, `_d`).

---

## Example End-to-End Pseudocode (Rebalance Iteration)

```python
def rebalance_at_date(t, cfg, data_store, prev_weights):

    # 1. Slice history
    hist = data_store.get_returns(start=t-cfg.lookback_days, end=t-1)

    # 2. Risk model
    Sigma_t, risk_diag = risk_engine.build_covariance(hist)

    # 3. Return views
    mu_ff, var_ff, ff_diag = ff_view.forecast(hist, factors_hist, cfg)
    mu_var, var_var, var_diag = var_view.forecast(hist, cfg)

    # 4. Build BL prior
    w_mkt = data_store.get_mkt_weights(t)
    pi = bl.reverse_optimization(Sigma_t, w_mkt, lambda_risk(cfg, hist))

    # 5. Stack views
    P, Q, Omega = view_builder.stack([
        {'mu': mu_ff, 'var': var_ff},
        {'mu': mu_var, 'var': var_var}
    ])

    mu_bl, Sigma_post = bl.posterior(pi, Sigma_t, P, Q, Omega, tau=cfg.tau)

    # 6. Optimize weights
    w_new, opt_diag = optimizer.solve(mu_bl, Sigma_t, constraints=cfg.constraints,
                                      prev_weights=prev_weights, costs=cfg.costs_bps)

    return w_new, {
        'risk': risk_diag,
        'ff': ff_diag,
        'var': var_diag,
        'bl': {'pi': pi, 'mu_bl': mu_bl},
        'opt': opt_diag
    }
```

---

## Example Markdown Report Skeleton (Auto-Generated)

**Run ID:** 2025-07-17T1830Z
**Universe Size:** 48 active (avg)
**Period:** 2012-01-01 → 2025-01-01
**Rebalance:** Monthly
**Lookback:** 5y rolling
**BL τ:** 0.05 | Factor conf=1.0 | VAR conf=2.0
**Costs:** 10 bps per side

### 1. Summary Table

| Metric | Strategy | IBOV  | Excess vs. IBOV |
| ------ | -------- | ----- | --------------- |
| CAGR   | 12.4%    | 8.3%  | +4.1%           |
| Vol    | 17.0%    | 23.5% | -6.5%           |
| Sharpe | 0.58     | 0.28  | +0.30           |
| Max DD | -32%     | -51%  | +19%            |

### 2. Equity Curve Plot

(figure)

### 3. Rolling Sharpe (36m)

(figure)

### 4. Weight Evolution Heatmap

(figure)

### 5. BL Diagnostics: Prior vs. Posterior

(figure + commentary)

### 6. Factor Attribution Regression

(table; α, βs)

### 7. View Forecast Skill

(IC, Hit rate)

### 8. Risk Model Stability

(Avg corr, vol spikes)

---

## Final Notes for PoC Execution

1. **Get the data right first.** No modeling patch will fix contaminated data.
2. **Reproduce IBOV correctly**; if benchmark wrong, performance claims meaningless.
3. **Log everything**: parameters, data version, model warnings.
4. **Start with no BL (pure market)**, then add one view at a time to confirm incremental impact.
5. **Always check units** when combining factor premia (annual) with VAR (daily).
6. **Plot the implied portfolio from π** before introducing views; sanity anchor.
7. **Monitor turnover**; if extreme, shrink Ω or increase costs.

---

### Quick Launch Checklist (Minimal Working Example)

* [ ] Download 5 tickers (PETR4.SA, VALE3.SA, ITUB4.SA, ABEV3.SA, BBDC4.SA) + ^BVSP + SELIC (2010→present).
* [ ] Construct daily log returns, excess vs. SELIC.
* [ ] Hardcode mock fundamentals (size, B/M) for first run; build dummy SMB/HML.
* [ ] Fit GARCH-only risk; skip DCC initially (use sample corr).
* [ ] Compute factor μ simple (β × 5% market premia placeholder).
* [ ] Run BL w/ 1 view; optimize; produce 3 rebalance steps.
* [ ] Confirm pipeline runs end-to-end; then layer realism.

---

## Concluding Directive

This document defines the **authoritative blueprint** for building, validating, and extending a **modular, Bayesian-integrated, risk-aware portfolio allocation research system for the Brazilian equity market**. All future design discussions, code implementations, and validation notebooks must reference section numbers in this specification. When making changes, update this document first, then the codebase, and finally re-run the regression test suites to ensure reproducibility.

Save this file as:
`/b3_alloc_system/docs/project_specification.md`

(Version-control every revision; tag releases aligned with major milestones.)

---

**End of Specification**
