from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_datareader
import yfinance as yf
from pandas_datareader import data as web


# ==================================================
# 1. PROJECT FOLDERS
# ==================================================

# This file is inside the "scripts" folder.
# .parent gives scripts, and .parent.parent gives the project folder.
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

RAW_FOLDER = PROJECT_FOLDER / "data" / "raw"
PROCESSED_FOLDER = PROJECT_FOLDER / "data" / "processed"

# Create the folders automatically if they do not already exist.
RAW_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. RESEARCH SETTINGS
# ==================================================

START_DATE = "2007-01-01"

# yfinance treats the end date as exclusive.
# Using 2026-01-01 allows us to collect data through December 2025.
DOWNLOAD_END_DATE = "2026-06-01"
RESEARCH_END_DATE = "2025-12-31"

TICKERS = [
    "SPY",  # U.S. equities
    "VEA",  # Developed international equities
    "IEF",  # Intermediate U.S. Treasury bonds
    "LQD",  # Investment-grade corporate bonds
    "VNQ",  # U.S. real estate
    "GLD",  # Gold
    "BIL",  # Treasury bills / cash proxy
]

FRED_SERIES = [
    "CPIAUCSL",  # Consumer Price Index
    "FEDFUNDS",  # Effective federal funds rate
    "USREC",     # NBER recession indicator
]


# ==================================================
# 3. DOWNLOAD ETF DATA
# ==================================================

print("=" * 60)
print("DOWNLOADING ETF MARKET DATA")
print("=" * 60)

market_data = yf.download(
    tickers=TICKERS,
    start=START_DATE,
    end=DOWNLOAD_END_DATE,
    auto_adjust=True,
    progress=False,
    group_by="column",
)

if market_data.empty:
    raise RuntimeError(
        "No ETF data was downloaded. Check the internet connection."
    )

# With auto_adjust=True, the Close field contains adjusted prices.
try:
    daily_prices = market_data["Close"].copy()
except KeyError as error:
    raise RuntimeError(
        "The downloaded ETF data does not contain a Close field."
    ) from error

# Put the columns in the intended research order.
daily_prices = daily_prices.reindex(columns=TICKERS)

missing_tickers = [
    ticker
    for ticker in TICKERS
    if ticker not in daily_prices.columns
    or daily_prices[ticker].dropna().empty
]

if missing_tickers:
    raise RuntimeError(
        f"No usable data was returned for: {missing_tickers}"
    )

daily_prices.index.name = "date"

raw_etf_path = RAW_FOLDER / "etf_daily_adjusted_prices.csv"
daily_prices.to_csv(raw_etf_path)

print(f"ETF data saved to:\n{raw_etf_path}")
print(f"Daily observations: {len(daily_prices):,}")


# ==================================================
# 4. CONVERT ETF DATA TO MONTH-END
# ==================================================

monthly_prices = daily_prices.resample("ME").last()

monthly_prices.columns = [
    f"adj_price_{ticker}"
    for ticker in monthly_prices.columns
]

monthly_returns = monthly_prices.pct_change(fill_method=None)

monthly_returns.columns = [
    column.replace("adj_price_", "return_")
    for column in monthly_returns.columns
]


# ==================================================
# 5. DOWNLOAD FRED MACROECONOMIC DATA
# ==================================================

print("\n" + "=" * 60)
print("DOWNLOADING FRED MACROECONOMIC DATA")
print("=" * 60)

fred_data = web.DataReader(
    FRED_SERIES,
    "fred",
    START_DATE,
    DOWNLOAD_END_DATE,
)

if fred_data.empty:
    raise RuntimeError(
        "No FRED data was downloaded. Check the internet connection."
    )
print("\nCPI observations near the end of 2025:")
print(
    fred_data.loc[
        "2025-09-01":"2025-12-01",
        ["CPIAUCSL"]
    ]
)
fred_data.index.name = "date"

raw_fred_path = RAW_FOLDER / "fred_macroeconomic_data.csv"
fred_data.to_csv(raw_fred_path)

print(f"FRED data saved to:\n{raw_fred_path}")
print(f"Macroeconomic observations: {len(fred_data):,}")


# ==================================================
# 6. CONVERT FRED DATA TO MONTH-END
# ==================================================

monthly_macro = fred_data.resample("ME").last()

monthly_macro = monthly_macro.rename(
    columns={
        "CPIAUCSL": "cpi_index",
        "FEDFUNDS": "fedfunds",
        "USREC": "nber_recession",
    }
)

# Calculate the 12-month percentage change in CPI.
monthly_macro["infl_yoy"] = (
    monthly_macro["cpi_index"].pct_change(
        periods=12,
        fill_method=None,
    )
)

# Calculate the six-month change in the federal funds rate.
monthly_macro["policy_6m_change"] = (
    monthly_macro["fedfunds"].diff(periods=6)
)


# ==================================================
# 7. COMBINE THE DATA
# ==================================================

processed_data = pd.concat(
    [
        monthly_prices,
        monthly_returns,
        monthly_macro,
    ],
    axis=1,
)

processed_data = processed_data.loc[
    :pd.Timestamp(RESEARCH_END_DATE)
]

processed_data.index.name = "date"

processed_path = (
    PROCESSED_FOLDER
    / "monthly_research_dataset.csv"
)

processed_data.to_csv(processed_path)


# ==================================================
# 8. CREATE A DATA-RETRIEVAL LOG
# ==================================================

retrieval_time = datetime.now(timezone.utc).isoformat()

retrieval_log = f"""
Beyond the 60/40 Portfolio
Data Retrieval Log

Retrieval time in UTC: {retrieval_time}
Research start date: {START_DATE}
Research end date: {RESEARCH_END_DATE}

ETF tickers:
{", ".join(TICKERS)}

FRED series:
{", ".join(FRED_SERIES)}

Python packages:
pandas: {pd.__version__}
yfinance: {yf.__version__}
pandas-datareader: {pandas_datareader.__version__}

Raw ETF file:
{raw_etf_path}

Raw FRED file:
{raw_fred_path}

Processed monthly file:
{processed_path}
""".strip()

log_path = RAW_FOLDER / "data_retrieval_log.txt"
log_path.write_text(retrieval_log, encoding="utf-8")


# ==================================================
# 9. BASIC VALIDATION OUTPUT
# ==================================================

print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETED")
print("=" * 60)

print(f"Processed file saved to:\n{processed_path}")

print(
    f"\nFirst dataset date: "
    f"{processed_data.index.min().date()}"
)

print(
    f"Last dataset date: "
    f"{processed_data.index.max().date()}"
)

print(
    f"Number of monthly rows: "
    f"{len(processed_data):,}"
)

print("\nMissing values by column:")
print(processed_data.isna().sum())

print("\nFirst five rows:")
print(processed_data.head())

print("\nLast five rows:")
print(processed_data.tail())