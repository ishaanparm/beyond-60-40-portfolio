from pathlib import Path

import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

PROCESSED_FOLDER = PROJECT_FOLDER / "data" / "processed"
TABLES_FOLDER = PROJECT_FOLDER / "outputs" / "tables"

TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_FOLDER / "monthly_research_dataset.csv"
OUTPUT_FILE = PROCESSED_FOLDER / "validated_monthly_dataset.csv"

VALIDATION_REPORT_FILE = TABLES_FOLDER / "data_validation_report.txt"
MISSING_VALUES_FILE = TABLES_FOLDER / "missing_values_by_column.csv"
EXTREME_RETURNS_FILE = TABLES_FOLDER / "extreme_return_observations.csv"


# ==================================================
# 2. EXPECTED COLUMNS
# ==================================================

TICKERS = [
    "SPY",
    "VEA",
    "IEF",
    "LQD",
    "VNQ",
    "GLD",
    "BIL",
]

PRICE_COLUMNS = [
    f"adj_price_{ticker}"
    for ticker in TICKERS
]

RETURN_COLUMNS = [
    f"return_{ticker}"
    for ticker in TICKERS
]

MACRO_COLUMNS = [
    "cpi_index",
    "fedfunds",
    "nber_recession",
    "infl_yoy",
    "policy_6m_change",
]

EXPECTED_COLUMNS = (
    PRICE_COLUMNS
    + RETURN_COLUMNS
    + MACRO_COLUMNS
)


# ==================================================
# 3. LOAD THE DATASET
# ==================================================

print("=" * 60)
print("LOADING MONTHLY RESEARCH DATASET")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "The monthly research dataset was not found.\n"
        "Run 01_data_collection.py first."
    )

data = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"],
)

data = data.set_index("date")
data = data.sort_index()

print(f"Dataset loaded from:\n{INPUT_FILE}")
print(f"Rows loaded: {len(data):,}")
print(f"Columns loaded: {len(data.columns):,}")


# ==================================================
# 4. CHECK REQUIRED COLUMNS
# ==================================================

missing_columns = [
    column
    for column in EXPECTED_COLUMNS
    if column not in data.columns
]

if missing_columns:
    raise RuntimeError(
        "The following required columns are missing:\n"
        f"{missing_columns}"
    )

print("\nAll required columns are present.")


# ==================================================
# 5. CHECK DATES
# ==================================================

duplicate_dates = data.index[data.index.duplicated()].tolist()

if duplicate_dates:
    raise RuntimeError(
        "Duplicate dates were found:\n"
        f"{duplicate_dates}"
    )

if not data.index.is_monotonic_increasing:
    raise RuntimeError(
        "The dates are not in chronological order."
    )

expected_months = pd.date_range(
    start=data.index.min(),
    end=data.index.max(),
    freq="ME",
)

missing_months = expected_months.difference(data.index)

non_month_end_dates = [
    date
    for date in data.index
    if not date.is_month_end
]

print(f"Duplicate dates: {len(duplicate_dates)}")
print(f"Missing calendar months: {len(missing_months)}")
print(f"Non-month-end dates: {len(non_month_end_dates)}")


# ==================================================
# 6. CHECK NUMERIC DATA TYPES
# ==================================================

non_numeric_columns = []

for column in EXPECTED_COLUMNS:
    if not pd.api.types.is_numeric_dtype(data[column]):
        non_numeric_columns.append(column)

if non_numeric_columns:
    raise RuntimeError(
        "These columns are not numeric:\n"
        f"{non_numeric_columns}"
    )

print("\nAll financial and macroeconomic columns are numeric.")


# ==================================================
# 7. CHECK PRICES
# ==================================================

invalid_price_rows = []

for column in PRICE_COLUMNS:
    invalid_mask = (
        data[column].notna()
        & (data[column] <= 0)
    )

    for date in data.index[invalid_mask]:
        invalid_price_rows.append(
            {
                "date": date,
                "column": column,
                "value": data.loc[date, column],
            }
        )

invalid_prices = pd.DataFrame(invalid_price_rows)

if not invalid_prices.empty:
    invalid_prices.to_csv(
        TABLES_FOLDER / "invalid_price_observations.csv",
        index=False,
    )

    raise RuntimeError(
        "Zero or negative ETF prices were found. "
        "Review invalid_price_observations.csv."
    )

print("No zero or negative ETF prices were found.")


# ==================================================
# 8. CHECK RETURNS
# ==================================================

impossible_return_rows = []
extreme_return_rows = []

for column in RETURN_COLUMNS:
    # A simple return cannot be less than -100%.
    impossible_mask = (
        data[column].notna()
        & (data[column] <= -1.0)
    )

    for date in data.index[impossible_mask]:
        impossible_return_rows.append(
            {
                "date": date,
                "column": column,
                "return": data.loc[date, column],
            }
        )

    # Flag returns larger than 50% in absolute value.
    # These are not automatically wrong, but must be inspected.
    extreme_mask = (
        data[column].notna()
        & (data[column].abs() > 0.50)
    )

    for date in data.index[extreme_mask]:
        extreme_return_rows.append(
            {
                "date": date,
                "column": column,
                "return": data.loc[date, column],
            }
        )

impossible_returns = pd.DataFrame(impossible_return_rows)
extreme_returns = pd.DataFrame(extreme_return_rows)

if not impossible_returns.empty:
    impossible_returns.to_csv(
        TABLES_FOLDER / "impossible_return_observations.csv",
        index=False,
    )

    raise RuntimeError(
        "Returns of -100% or lower were found. "
        "Review impossible_return_observations.csv."
    )

if not extreme_returns.empty:
    extreme_returns.to_csv(
        EXTREME_RETURNS_FILE,
        index=False,
    )
else:
    pd.DataFrame(
        columns=["date", "column", "return"]
    ).to_csv(
        EXTREME_RETURNS_FILE,
        index=False,
    )

print("No mathematically impossible returns were found.")
print(
    "Returns exceeding 50% in absolute value: "
    f"{len(extreme_returns)}"
)


# ==================================================
# 9. MISSING-VALUE ANALYSIS
# ==================================================

missing_summary = pd.DataFrame(
    {
        "missing_count": data.isna().sum(),
        "total_rows": len(data),
    }
)

missing_summary["missing_percent"] = (
    missing_summary["missing_count"]
    / missing_summary["total_rows"]
    * 100
)

missing_summary.index.name = "column"

missing_summary.to_csv(MISSING_VALUES_FILE)

print("\nMissing-value summary:")
print(missing_summary)


# ==================================================
# 10. FIND COMMON ETF START DATE
# ==================================================

all_returns_available = data[RETURN_COLUMNS].notna().all(axis=1)

if not all_returns_available.any():
    raise RuntimeError(
        "There is no month in which all seven ETF "
        "returns are available."
    )

common_return_start = data.index[
    all_returns_available
][0]

print(
    "\nFirst month with all seven ETF returns: "
    f"{common_return_start.date()}"
)


# ==================================================
# 11. CREATE VALIDATED DATASET
# ==================================================

validated_data = data.loc[
    common_return_start:
].copy()

# Make certain there are no return gaps after the common start.
internal_return_gaps = (
    validated_data[RETURN_COLUMNS]
    .isna()
    .any(axis=1)
)

if internal_return_gaps.any():
    gap_dates = validated_data.index[
        internal_return_gaps
    ].tolist()

    raise RuntimeError(
        "Missing ETF returns exist after the common start date:\n"
        f"{gap_dates}"
    )

validated_data.to_csv(OUTPUT_FILE)

print(
    "\nValidated dataset saved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    "Validated dataset date range: "
    f"{validated_data.index.min().date()} through "
    f"{validated_data.index.max().date()}"
)

print(
    f"Validated monthly rows: {len(validated_data):,}"
)


# ==================================================
# 12. FIND MACROECONOMIC AVAILABILITY
# ==================================================

model_columns = (
    RETURN_COLUMNS
    + [
        "infl_yoy",
        "fedfunds",
        "policy_6m_change",
        "nber_recession",
    ]
)

all_model_fields_available = (
    validated_data[model_columns]
    .notna()
    .all(axis=1)
)

if all_model_fields_available.any():
    first_model_ready_date = validated_data.index[
        all_model_fields_available
    ][0]
else:
    first_model_ready_date = None

macro_missing_dates = {}

for column in MACRO_COLUMNS:
    dates = validated_data.index[
        validated_data[column].isna()
    ]

    macro_missing_dates[column] = [
        date.strftime("%Y-%m-%d")
        for date in dates
    ]


# ==================================================
# 13. RECESSION-INDICATOR CHECK
# ==================================================

recession_values = set(
    validated_data["nber_recession"]
    .dropna()
    .unique()
)

valid_recession_values = {0, 1, 0.0, 1.0}

unexpected_recession_values = (
    recession_values
    - valid_recession_values
)

if unexpected_recession_values:
    raise RuntimeError(
        "Unexpected values were found in nber_recession:\n"
        f"{unexpected_recession_values}"
    )

print(
    "\nNBER recession values found: "
    f"{sorted(recession_values)}"
)


# ==================================================
# 14. CREATE VALIDATION REPORT
# ==================================================

if first_model_ready_date is not None:
    model_ready_text = first_model_ready_date.strftime(
        "%Y-%m-%d"
    )
else:
    model_ready_text = "No fully complete model row found"

missing_months_text = (
    ", ".join(
        date.strftime("%Y-%m-%d")
        for date in missing_months
    )
    if len(missing_months) > 0
    else "None"
)

non_month_end_text = (
    ", ".join(
        date.strftime("%Y-%m-%d")
        for date in non_month_end_dates
    )
    if non_month_end_dates
    else "None"
)

macro_missing_text = "\n".join(
    f"{column}: "
    + (
        ", ".join(dates)
        if dates
        else "None"
    )
    for column, dates in macro_missing_dates.items()
)

report = f"""
Beyond the 60/40 Portfolio
Data Validation Report

INPUT DATA
Input file: {INPUT_FILE}
Original rows: {len(data)}
Original first date: {data.index.min().date()}
Original last date: {data.index.max().date()}

DATE VALIDATION
Duplicate dates: {len(duplicate_dates)}
Missing calendar months: {len(missing_months)}
Missing calendar-month dates: {missing_months_text}
Non-month-end dates: {len(non_month_end_dates)}
Non-month-end date values: {non_month_end_text}

COLUMN VALIDATION
Expected columns: {len(EXPECTED_COLUMNS)}
Missing required columns: {missing_columns}
Non-numeric required columns: {non_numeric_columns}

PRICE AND RETURN VALIDATION
Invalid zero or negative prices: {len(invalid_prices)}
Impossible returns of -100% or lower: {len(impossible_returns)}
Returns exceeding 50% in absolute value: {len(extreme_returns)}

COMMON SAMPLE
First month with all seven ETF returns: {common_return_start.date()}
Validated dataset rows: {len(validated_data)}
Validated dataset end date: {validated_data.index.max().date()}
First row with all model fields available: {model_ready_text}

MACROECONOMIC MISSING DATES
{macro_missing_text}

RECESSION INDICATOR
Observed values: {sorted(recession_values)}

OUTPUT FILE
{OUTPUT_FILE}

VALIDATION RESULT
The dataset passed the required structural, date, price,
return, and recession-indicator checks.
Missing macroeconomic values were documented but not
filled or replaced.
""".strip()

VALIDATION_REPORT_FILE.write_text(
    report,
    encoding="utf-8",
)


# ==================================================
# 15. FINAL OUTPUT
# ==================================================

print("\n" + "=" * 60)
print("DATA VALIDATION COMPLETED")
print("=" * 60)

print(
    f"Validation report saved to:\n"
    f"{VALIDATION_REPORT_FILE}"
)

print(
    f"\nMissing-value table saved to:\n"
    f"{MISSING_VALUES_FILE}"
)

print(
    f"\nExtreme-return review saved to:\n"
    f"{EXTREME_RETURNS_FILE}"
)

print("\nFirst five validated rows:")
print(validated_data.head())

print("\nLast five validated rows:")
print(validated_data.tail())