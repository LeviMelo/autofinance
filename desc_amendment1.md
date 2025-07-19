# Amendment 1 – Critical Specification Updates

**To be appended alongside** `/b3_alloc_system/docs/project_specification.md`
**Effective Date:** 18 July 2025

> **Purpose** – This amendment incorporates four non‑negotiable requirements that emerged from user review. All sections below **replace or extend** the corresponding items in the original specification. Where a paragraph begins with **“UPDATED:”** the earlier text is overridden. Where a paragraph begins with **“NEW:”** the text introduces completely new functionality.

---

## Table of Contents

- [Amendment 1 – Critical Specification Updates](#amendment1--critical-specification-updates)
  - [Table of Contents](#table-of-contents)
  - [Overview of Amendment Scope](#overview-of-amendment-scope)
  - [Requirement ① – Actionable Trade Generation (Integer Share Lots)](#requirement--actionable-trade-generation-integer-share-lots)
    - [1.1 Pipeline Placement](#11-pipeline-placement)
    - [1.2 NEW: `trade_calculator` Module](#12-new-trade_calculator-module)
    - [1.3 UPDATED: Backtest Execution Logic](#13-updated-backtest-execution-logic)
    - [1.4 Data \& Storage](#14-data--storage)
  - [Requirement ② – Expanded Asset Universe \& FX Risk Modeling](#requirement--expanded-asset-universe--fx-risk-modeling)
    - [2.1 UPDATED: Universe Definition](#21-updated-universe-definition)
    - [2.2 NEW: FX Data Ingestion](#22-new-fx-data-ingestion)
    - [2.3 UPDATED: Factor Construction](#23-updated-factor-construction)
    - [2.4 UPDATED: Risk Engine](#24-updated-risk-engine)
    - [2.5 Configuration](#25-configuration)
  - [Requirement ③ – Brazilian Tax Law Integration \& DARF Generation](#requirement--brazilian-tax-law-integration--darf-generation)
    - [3.1 NEW Module Group](#31-new-module-group)
    - [3.2 Persistent Ledgers](#32-persistent-ledgers)
    - [3.3 Tax Logic Rules Implemented](#33-tax-logic-rules-implemented)
    - [3.4 NEW: `tax_tracker`](#34-new-tax_tracker)
    - [3.5 NEW: `darf_reporter`](#35-new-darf_reporter)
  - [Requirement ④ – Multi‑Portfolio Support \& Qualitative View Input](#requirement--multiportfolio-support--qualitative-view-input)
    - [4.1 UPDATED: Configuration System](#41-updated-configuration-system)
    - [4.2 NEW: `views_parser`](#42-new-views_parser)
    - [4.3 UPDATED: Black‑Litterman Module](#43-updated-blacklitterman-module)
    - [4.4 CLI / Notebook Helper](#44-cli--notebook-helper)
  - [Repository \& Module Additions](#repository--module-additions)
  - [Backwards‑Compatibility Notes](#backwardscompatibility-notes)
  - [Revised Development Timeline](#revised-development-timeline)
  - [Closing Remark](#closing-remark)

---

## Overview of Amendment Scope

| Requirement                         | New Modules                                           | Key Data Additions             | Risk to Deadline | Mitigations                              |
| ----------------------------------- | ----------------------------------------------------- | ------------------------------ | ---------------- | ---------------------------------------- |
| Trade generation                    | `trade_calculator`, `order_formatter`                 | realtime quotes snapshot       | **Medium**       | isolate trade logic; mock prices for dev |
| FX & wider universe                 | FX factor ingestion, FX beta in risk & return engines | `usdbrl_daily`, BDR metadata   | **Low**          | incremental; VAR & BL already modular    |
| Tax awareness & DARF                | `tax_tracker`, `darf_reporter`                        | cost‑basis ledger, sale ledger | **High**         | start with stock‑only logic, iterate     |
| Multi‑portfolio & qualitative views | hierarchical configs, `views_parser`                  | per‑portfolio YAML, view JSON  | **Medium**       | strict schema validation via pydantic    |

---

## Requirement ① – Actionable Trade Generation (Integer Share Lots)

### 1.1 Pipeline Placement

```
Optimizer  ──► Trade Calculator  ──► Order Formatter  ──► Execution Simulator
                         ▲
                 historical positions
```

### 1.2 NEW: `trade_calculator` Module

| Function                                             | Signature                                          | Description                                                |
| ---------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------- |
| `calculate_target_shares(target_w, pv, last_prices)` | returns `DataFrame(ticker, tgt_shares, tgt_value)` | Converts optimizer weights into integer share counts.      |
| `resolve_fractional(tgt_shares, mode='round')`       | returns `DataFrame`                                | Rounds fractional shares. Modes: `round`, `floor`, `ceil`. |
| `compute_trade_list(current_shares, tgt_shares)`     | returns `DataFrame(ticker, delta_shares, action)`  | Generates **BUY / SELL / HOLD** directives.                |

*All calculations assume access to **closing prices at T₀** (same prices used to evaluate optimizer objective).*

### 1.3 UPDATED: Backtest Execution Logic

Section **11.3 Portfolio Holding Period Return Realization** is expanded:

* After each rebalance the simulator executes the integer trade list at the **price snapshot already used by the optimizer**. Residual cash (due to rounding) is carried at the risk‑free rate.

### 1.4 Data & Storage

**`positions_ledger`** table (new):

| date | ticker | shares | price | value | cash\_post\_trade |
| ---- | ------ | ------ | ----- | ----- | ----------------- |

Maintains end‑of‑day positions for cost‑basis tracking (see Requirement ③).

---

## Requirement ② – Expanded Asset Universe & FX Risk Modeling

### 2.1 UPDATED: Universe Definition

*Universe may now contain*:

* **Domestic equities** (ON/PN)
* **BDRs** (`.SA` tickers ending in 34, 35, etc.)
* **ETFs** (e.g., BOVA11.SA)
* **FII/REIT‑like tickers** (optional future extension)

### 2.2 NEW: FX Data Ingestion

*Add to* **4.1 Data Sources**

| Data Type         | Frequency | Source             |
| ----------------- | --------- | ------------------ |
| USD/BRL spot rate | Daily     | BCB SGS series 1 ( |

"`cotação de fechamento PTAX`") |

**`usdbrl_daily`** table mirrors `risk_free_daily`.

### 2.3 UPDATED: Factor Construction

*Section 8 (Return Engine)*

* **NEW FX Factor (FX):** daily log return of USD/BRL.
* For **BDRs** and any ticker flagged `fx_sensitive=True`, run rolling regressions against `[MKT, SMB, HML, FX]`.
* Non‑FX assets may still include FX factor but expected beta ≈ 0.

### 2.4 UPDATED: Risk Engine

*Covariance matrix Σ must include the USD/BRL series* (treated as an additional “asset” with zero weight in optimizer but feeding correlations). Implementation:

1. Append FX return column to returns panel.
2. Fit GARCH + DCC including FX column.
3. When building Σ for optimizer, **drop** FX row/col but **store** conditional FX beta for downstream attribution.

### 2.5 Configuration

*`config/*.yaml` gains*

```yaml
universe:
  include_fx_factor: true
  fx_series: USD_BRL
  asset_flags:
    PETR4.SA: {fx_sensitive: false}
    AAPL34.SA: {fx_sensitive: true}
```

---

## Requirement ③ – Brazilian Tax Law Integration & DARF Generation

### 3.1 NEW Module Group

```
positions_ledger ─► tax_tracker ─► rebalance_decision
                               └─► darf_reporter
```

### 3.2 Persistent Ledgers

**`positions_ledger`** (see 1.4) now records *`lot_id`*, linking buys for FIFO/PM cost.

**`sales_ledger`** (new):

\| sale\_id | date | ticker | shares\_sold | gross\_value | cost\_basis | gain | asset\_class | darf\_exempt\_flag |

### 3.3 Tax Logic Rules Implemented

| Asset Class        | Exemption                                 | Tax Rate         | Deductible Costs     |
| ------------------ | ----------------------------------------- | ---------------- | -------------------- |
| **Stocks (ON/PN)** | Monthly gross sale ≤ R\$ 20 000 → **0 %** | 15 % on net gain | Broker fees, B3 fees |
| **BDRs**           | **No exemption**                          | 15 % on net gain | idem                 |
| **ETFs**           | **No exemption**                          | 15 % on net gain | idem                 |

*Losses can offset gains **within same asset class**.*

### 3.4 NEW: `tax_tracker`

1. **`update_ledgers(trade_df)`** – after every rebalance populate positions & sales.
2. **`compute_mensal_tax(month)`** – iterate sales\_ledger; apply rules; return `tax_due`.
3. **API to Rebalance Engine:** returns *expected after‑tax impact* if a candidate sell is executed; fed into optimizer **cost term**:

$$
\text{adj\_cost}_i = \text{broker\_cost}_i + \frac{\text{expected tax liability}_i}{\text{portfolio value}}
$$

### 3.5 NEW: `darf_reporter`

*Monthly cron job.*

Output:

* Excel (`.xlsx`) sheet with: `month`, `asset_class`, `gross_sales`, `net_gain`, `tax_due`, `DARF_code` (6015).
* Optional PDF summary via `python-docx` → `.pdf`.

---

## Requirement ④ – Multi‑Portfolio Support & Qualitative View Input

### 4.1 UPDATED: Configuration System

*Top‑level directory* now:

```
config/
   portfolio_A.yaml
   portfolio_B.yaml
   views/
      qualitative_views_A.yaml
      qualitative_views_B.yaml
```

**`portfolio_*.yaml`** schema (excerpt):

```yaml
meta:
  portfolio_name: Port_A
  base_currency: BRL
data:
  tickers: [PETR4.SA, VALE3.SA, AAPL34.SA]
risk:
  name_cap: 0.08
bl:
  tau: 0.05
  qualitative_views_file: views/qualitative_views_A.yaml
tax:
  enable: true
  brokerage_fee_bps: 5
```

### 4.2 NEW: `views_parser`

Expected qualitative file format:

```yaml
views:
  - type: relative
    expr: "AAPL34.SA – VALE3.SA"
    magnitude: 0.03    # +3 % expected
    confidence: 0.70
  - type: absolute
    expr: "PETR4.SA"
    magnitude: 0.02
    confidence: 0.50
```

Parser translates:

1. **`expr`** → parse left/right tickers; build P row(s).
2. **`magnitude`** → Q entry; sign handled from expression.
3. **`confidence`** → variance term via mapping

   $$
   \sigma_{\text{view}}^2 = \frac{1}{\text{confidence} \times k}
   $$

   where *k* configurable (default = 100).

### 4.3 UPDATED: Black‑Litterman Module

*Add step* **“user\_views”** after model views; stack:

```
P_total = [P_FF
           P_VAR
           P_user]
Q_total = [...]
Ω_total = blkdiag(Ω_FF, Ω_VAR, Ω_user)
```

### 4.4 CLI / Notebook Helper

Function `create_qual_view(expr, magnitude, conf)` returns YAML snippet; embedded in Jupyter for ease.

---

## Repository & Module Additions

```
src/b3alloc/
   trades/
      trade_calculator.py    # Req ①
      order_formatter.py
   fx/
      fx_ingest.py           # Req ②
   taxes/
      ledger.py              # Req ③
      tax_tracker.py
      darf_reporter.py
   views/
      views_parser.py        # Req ④
tests/
   test_trade_calculator.py
   test_fx_factor.py
   test_tax_tracker.py
   test_views_parser.py
```

---

## Backwards‑Compatibility Notes

* Running the pipeline **without** the new YAML keys keeps default behaviour (weights only, no tax, single portfolio).
* `trade_calculator` can be stubbed (`round=False`) to reproduce old continuous‑weight backtests.
* Tax modules activate only when `tax.enable=true`.

---

## Revised Development Timeline

| Week     | Milestone Additions                                                     |
| -------- | ----------------------------------------------------------------------- |
| **8**    | Design ledgers; implement `positions_ledger`; stub tax cost as 0 %      |
| **9–10** | Build `trade_calculator`, integrate with backtest; adjust unit tests    |
| **11**   | FX ingestion; add FX factor to regressions; update risk engine          |
| **12**   | Implement `tax_tracker` logic for stocks (R\$ 20 k rule)                |
| **13**   | Extend tax logic to BDRs & ETFs; generate DARF prototype                |
| **14**   | Qualitative view parser & BL integration; multi‑portfolio config loader |
| **15**   | End‑to‑end regression test on two portfolios; freeze v0.2 tag           |

---

## Closing Remark

All developers must **re‑base their feature branches onto this amendment** before further commits. Implementation PRs **must reference the numbered sections in this document** and include unit/integration tests proving compliance with the new requirements.

**— End of Amendment 1**
