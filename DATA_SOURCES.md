# Data Sources

This document describes the data sources, retrieval details, transformations,
and known data limitations for Working Paper 1.0.

## Archived Research Release

- Retrieval timestamp: July 17, 2026 at 02:28:20 UTC
- Raw retrieval window: January 1, 2007 through December 31, 2025
- Main monthly research sample: January 2008 through December 2025
- Out-of-sample comparison: January 2013 through December 2025
- Portfolio evaluation frequency: Monthly
- Standard rebalancing frequency: Quarterly

The published paper and archived result files correspond to this retrieval.
A later download may differ because third party providers can revise historical
adjusted price data.

## ETF Market Data

ETF market data were retrieved from Yahoo Finance through `yfinance` version
1.5.1 with automatic price adjustment enabled.

| Ticker | Exposure |
|---|---|
| SPY | U.S. equities |
| VEA | Developed markets outside the United States |
| IEF | Intermediate-term U.S. Treasury bonds |
| LQD | Investment-grade corporate bonds |
| VNQ | Listed U.S. real estate |
| GLD | Gold |
| BIL | Treasury bills and cash-equivalent exposure |

Daily adjusted prices were converted to month end observations. Simple monthly
total returns were then calculated from consecutive adjusted month-end prices.

## Macroeconomic Data

Macroeconomic series were retrieved from the Federal Reserve Economic Data
database using `pandas-datareader` version 0.11.1.

| Series | Description | Research use |
|---|---|---|
| CPIAUCSL | Consumer Price Index for All Urban Consumers | Inflation trend and real-return calculation |
| FEDFUNDS | Effective federal funds rate | Monetary-policy classification |
| USREC | NBER-based recession indicator | Recession-month stress analysis |

Macroeconomic signals were lagged by two months to reduce look-ahead bias.

## Known Missing Observations

The raw retrieval begins in January 2007. VEA has six initial missing
month end observations and BIL has four initial missing observations because
complete histories were not yet available at the beginning of the raw window.

These observations occur before the January 2008 research sample and do not
represent deleted observations.

A complete October 2025 CPI observation was unavailable. The regime strategy
retained the most recently available inflation observation rather than
interpolating an unobserved CPI index value. Real CAGR was calculated using the
available beginning and ending CPI index levels.

## Data Availability

Raw and processed third party datasets are not redistributed in this public
repository.

The following files remain local:

```text
data/raw/etf_daily_adjusted_prices.csv
data/raw/fred_macroeconomic_data.csv
data/processed/monthly_research_dataset.csv
data/processed/validated_monthly_dataset.csv

```

The public repository includes the data collection and validation scripts,
sanitized retrieval metadata, derived portfolio results, final tables, and
figures.

## Authoritative Results

Files beginning with `fair_comparison_` contain the authoritative six portfolio
results reported in the working paper.

Files beginning with `baseline_` and `optimized_` are intermediate audit
outputs.

## Software Environment

- pandas 3.0.3
- NumPy 2.5.1
- SciPy 1.18.0
- matplotlib 3.11.0
- yfinance 1.5.1
- pandas-datareader 0.11.1
- CVXPY 1.9.2
- openpyxl 3.1.5

The complete dependency environment is recorded in `requirements.txt`.


- All six analysis scripts are present.      
- Both final paper files exist; the PDF contains 20 pages.  
- The authoritative six-portfolio monthly results are present. 
- The final performance workbook and both final figures are present.   
- The retrieval log uses relative paths and records the timestamp and package versions. 
- `.gitignore` correctly excludes virtual environments and private dataset CSVs. 
- The code and research-material license is present and correctly scoped. 
- I directly checked all four private raw/processed dataset paths; none is publicly uploaded.
- Searches found no `password`, `api_key`, `secret`, `token`, or `C:\Users` matches.


