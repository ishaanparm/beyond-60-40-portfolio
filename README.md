# Beyond the 60/40 Portfolio

Independent investment research working paper by Ishaan Parmar.

## Research Question

Can a transparent, long only, regime sensitive multi-asset portfolio improve downside protection and risk-adjusted performance relative to a traditional 60/40 portfolio?

## Main Findings

- The traditional 60/40 portfolio produced the highest nominal and inflation adjusted growth.
- Constrained mean variance optimization produced the highest Sharpe and Sortino ratios.
- Minimum-CVaR allocation provided the strongest tail-risk protection but substantially reduced growth.
- The regime-sensitive strategy reduced downside risk but failed the precommitted overall success standard.

## Sample and Methodology

- Market-data period: January 2008 through December 2025
- Out-of-sample test: January 2013 through December 2025
- Seven ETF proxies: SPY, VEA, IEF, LQD, VNQ, GLD, and BIL
- Quarterly rebalancing
- Long-only, unlevered portfolios
- Transaction costs and robustness tests included
- Lagged inflation and monetary-policy signals used to avoid look-ahead bias

## Repository Structure

- `scripts/` — complete Python analysis pipeline
- `paper/` — working paper in PDF and Word formats
- `outputs/tables/` — final tables and diagnostics
- `outputs/figures/` — final charts
- `metadata/` — sanitized data-retrieval information
- `DATA_SOURCES.md` — data-source and transformation documentation
- `requirements.txt` — pinned Python environment

## Python Environment

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```
