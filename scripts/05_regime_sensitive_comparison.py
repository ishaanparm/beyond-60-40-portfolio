from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

PROCESSED_FOLDER = PROJECT_FOLDER / "data" / "processed"
TABLES_FOLDER = PROJECT_FOLDER / "outputs" / "tables"
FIGURES_FOLDER = PROJECT_FOLDER / "outputs" / "figures"

TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

DATA_FILE = (
    PROCESSED_FOLDER
    / "validated_monthly_dataset.csv"
)

OPTIMIZED_FILE = (
    TABLES_FOLDER
    / "optimized_monthly_results.csv"
)

MONTHLY_OUTPUT_FILE = (
    TABLES_FOLDER
    / "fair_comparison_monthly_results.csv"
)

METRICS_CSV_FILE = (
    TABLES_FOLDER
    / "fair_comparison_performance_metrics.csv"
)

METRICS_EXCEL_FILE = (
    TABLES_FOLDER
    / "fair_comparison_performance_metrics.xlsx"
)

ANNUAL_RETURNS_FILE = (
    TABLES_FOLDER
    / "fair_comparison_annual_returns.xlsx"
)

SIGNALS_FILE = (
    TABLES_FOLDER
    / "regime_signal_history.csv"
)

ALLOCATIONS_FILE = (
    TABLES_FOLDER
    / "regime_rebalance_allocations.csv"
)

REGIME_RESULTS_FILE = (
    TABLES_FOLDER
    / "performance_by_regime.xlsx"
)

WEALTH_FIGURE = (
    FIGURES_FOLDER
    / "fair_comparison_cumulative_wealth.png"
)

DRAWDOWN_FIGURE = (
    FIGURES_FOLDER
    / "fair_comparison_drawdowns.png"
)


# ==================================================
# 2. RESEARCH SETTINGS
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

RETURN_COLUMNS = [
    f"return_{ticker}"
    for ticker in TICKERS
]

TEST_START_DATE = pd.Timestamp(
    "2013-01-31"
)

REBALANCE_MONTHS = {
    1,
    4,
    7,
    10,
}

TRANSACTION_COST_RATE = 0.001

MONTHS_PER_YEAR = 12

# A return dated January 31 is treated as the
# return earned during January.
#
# Therefore, the model conservatively uses
# macroeconomic data no later than November.
SIGNAL_LAG_MONTHS = 2

TREND_LOOKBACK_MONTHS = 6


# ==================================================
# 3. FIXED PORTFOLIO ALLOCATIONS
# ==================================================

FIXED_PORTFOLIOS = {
    "Traditional 60/40": {
        "SPY": 0.60,
        "VEA": 0.00,
        "IEF": 0.40,
        "LQD": 0.00,
        "VNQ": 0.00,
        "GLD": 0.00,
        "BIL": 0.00,
    },

    "Strategic Diversified": {
        "SPY": 0.35,
        "VEA": 0.15,
        "IEF": 0.15,
        "LQD": 0.10,
        "VNQ": 0.10,
        "GLD": 0.10,
        "BIL": 0.05,
    },
}


# ==================================================
# 4. PREDETERMINED REGIME ALLOCATIONS
# ==================================================

REGIME_WEIGHTS = {
    "Rising inflation + Tightening": {
        "SPY": 0.20,
        "VEA": 0.05,
        "IEF": 0.10,
        "LQD": 0.10,
        "VNQ": 0.10,
        "GLD": 0.20,
        "BIL": 0.25,
    },

    "Rising inflation + Easing": {
        "SPY": 0.30,
        "VEA": 0.10,
        "IEF": 0.10,
        "LQD": 0.10,
        "VNQ": 0.15,
        "GLD": 0.15,
        "BIL": 0.10,
    },

    "Falling inflation + Tightening": {
        "SPY": 0.25,
        "VEA": 0.10,
        "IEF": 0.20,
        "LQD": 0.15,
        "VNQ": 0.05,
        "GLD": 0.05,
        "BIL": 0.20,
    },

    "Falling inflation + Easing": {
        "SPY": 0.35,
        "VEA": 0.15,
        "IEF": 0.20,
        "LQD": 0.10,
        "VNQ": 0.10,
        "GLD": 0.05,
        "BIL": 0.05,
    },
}


# ==================================================
# 5. VALIDATE WEIGHTS
# ==================================================

def validate_basic_weights(
    portfolio_name,
    weight_dictionary,
):
    weights = pd.Series(
        weight_dictionary,
        index=TICKERS,
        dtype=float,
    )

    if weights.isna().any():
        raise RuntimeError(
            f"{portfolio_name} is missing weights."
        )

    if (weights < 0).any():
        raise RuntimeError(
            f"{portfolio_name} contains negative weights."
        )

    if not np.isclose(
        weights.sum(),
        1.0,
    ):
        raise RuntimeError(
            f"{portfolio_name} weights sum to "
            f"{weights.sum():.8f}, not 1.0."
        )


def validate_regime_weights(
    regime_name,
    weight_dictionary,
):
    validate_basic_weights(
        regime_name,
        weight_dictionary,
    )

    weights = pd.Series(
        weight_dictionary,
        index=TICKERS,
        dtype=float,
    )

    if (weights > 0.40 + 1e-10).any():
        raise RuntimeError(
            f"{regime_name} exceeds the "
            "40% individual-asset limit."
        )

    total_equities = (
        weights["SPY"]
        + weights["VEA"]
    )

    total_real_assets = (
        weights["VNQ"]
        + weights["GLD"]
    )

    if total_equities > 0.70 + 1e-10:
        raise RuntimeError(
            f"{regime_name} exceeds the "
            "70% equity limit."
        )

    if total_real_assets > 0.30 + 1e-10:
        raise RuntimeError(
            f"{regime_name} exceeds the "
            "30% real-asset limit."
        )


for name, weights in FIXED_PORTFOLIOS.items():
    validate_basic_weights(
        name,
        weights,
    )

for name, weights in REGIME_WEIGHTS.items():
    validate_regime_weights(
        name,
        weights,
    )


# ==================================================
# 6. LOAD DATA
# ==================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "validated_monthly_dataset.csv was not found.\n"
        "Run 02_data_validation.py first."
    )

if not OPTIMIZED_FILE.exists():
    raise FileNotFoundError(
        "optimized_monthly_results.csv was not found.\n"
        "Run 04_optimized_portfolios.py first."
    )

data = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"],
)

data = data.set_index("date")
data = data.sort_index()

required_columns = (
    RETURN_COLUMNS
    + [
        "cpi_index",
        "infl_yoy",
        "fedfunds",
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise RuntimeError(
        "Required columns are missing:\n"
        f"{missing_columns}"
    )

asset_returns = data[
    RETURN_COLUMNS
].copy()

asset_returns.columns = TICKERS

test_returns = asset_returns.loc[
    TEST_START_DATE:
].copy()

if test_returns.isna().any().any():
    raise RuntimeError(
        "Missing asset returns exist "
        "during the test period."
    )

# ==================================================
# CUMULATIVE INFLATION FOR REAL CAGR
# ==================================================

# Portfolio returns begin in January 2013, so the
# starting price level is December 2012.
START_CPI_DATE = (
    TEST_START_DATE
    - pd.offsets.MonthEnd(1)
)

END_CPI_DATE = (
    test_returns.index.max()
)

start_cpi = data.loc[
    START_CPI_DATE,
    "cpi_index",
]

end_cpi = data.loc[
    END_CPI_DATE,
    "cpi_index",
]

if pd.isna(start_cpi) or pd.isna(end_cpi):
    raise RuntimeError(
        "The beginning or ending CPI level "
        "required for real CAGR is missing."
    )

CUMULATIVE_INFLATION_FACTOR = float(
    end_cpi / start_cpi
)

print(
    "Cumulative inflation factor: "
    f"{CUMULATIVE_INFLATION_FACTOR:.4f}"
)

print("=" * 72)
print("REGIME-SENSITIVE AND FAIR-COMPARISON ANALYSIS")
print("=" * 72)

print(
    "Test period: "
    f"{test_returns.index.min().date()} through "
    f"{test_returns.index.max().date()}"
)

print(
    f"Test months: {len(test_returns)}"
)


# ==================================================
# 7. CREATE LAGGED MACROECONOMIC SIGNALS
# ==================================================

signals = pd.DataFrame(
    index=data.index
)

# Record the dates of genuinely available inflation data.
inflation_source_dates = pd.Series(
    data.index,
    index=data.index,
).where(
    data["infl_yoy"].notna()
)

# Use lagged information only. When no new inflation
# observation exists, retain the latest available release.
signals["signal_source_date"] = (
    inflation_source_dates
    .shift(SIGNAL_LAG_MONTHS)
    .ffill()
)

signals["inflation_signal"] = (
    data["infl_yoy"]
    .shift(SIGNAL_LAG_MONTHS)
    .ffill()
)

signals["inflation_6m_change"] = (
    signals["inflation_signal"]
    .diff(TREND_LOOKBACK_MONTHS)
)

signals["fedfunds_signal"] = (
    data["fedfunds"]
    .shift(SIGNAL_LAG_MONTHS)
)

signals["policy_6m_change"] = (
    signals["fedfunds_signal"]
    .diff(TREND_LOOKBACK_MONTHS)
)


def classify_direction(
    change_series,
    positive_label,
    negative_label,
):
    """
    Classify the sign of a six-month change.

    When the change equals zero, retain the
    previous nonzero state.
    """

    states = []
    previous_state = None

    for value in change_series:
        if pd.isna(value):
            states.append(pd.NA)
            continue

        if value > 0:
            previous_state = positive_label

        elif value < 0:
            previous_state = negative_label

        elif previous_state is None:
            previous_state = negative_label

        states.append(previous_state)

    return pd.Series(
        states,
        index=change_series.index,
        dtype="object",
    )


signals["inflation_trend"] = classify_direction(
    signals["inflation_6m_change"],
    "Rising inflation",
    "Falling inflation",
)

signals["policy_state"] = classify_direction(
    signals["policy_6m_change"],
    "Tightening",
    "Easing",
)

signals["regime"] = [
    (
        f"{inflation_state} + {policy_state}"
        if (
            pd.notna(inflation_state)
            and pd.notna(policy_state)
        )
        else pd.NA
    )
    for inflation_state, policy_state
    in zip(
        signals["inflation_trend"],
        signals["policy_state"],
    )
]

test_signals = signals.reindex(
    test_returns.index
).copy()

if test_signals["regime"].isna().any():
    missing_signal_dates = (
        test_signals.index[
            test_signals["regime"].isna()
        ].tolist()
    )

    raise RuntimeError(
        "Missing regime signals exist:\n"
        f"{missing_signal_dates}"
    )

unknown_regimes = sorted(
    set(
        test_signals["regime"].unique()
    )
    - set(REGIME_WEIGHTS)
)

if unknown_regimes:
    raise RuntimeError(
        "Unknown regime labels were created:\n"
        f"{unknown_regimes}"
    )

test_signals.to_csv(
    SIGNALS_FILE
)


# ==================================================
# 8. CREATE QUARTERLY TARGET WEIGHTS
# ==================================================

rebalance_dates = [
    date
    for date in test_returns.index
    if date.month in REBALANCE_MONTHS
]

if (
    test_returns.index.min()
    not in rebalance_dates
):
    raise RuntimeError(
        "The test period must begin "
        "in a rebalance month."
    )


def create_fixed_target_map(
    weight_dictionary,
):
    target = pd.Series(
        weight_dictionary,
        index=TICKERS,
        dtype=float,
    )

    return {
        date: target.copy()
        for date in rebalance_dates
    }


fixed_target_maps = {
    name: create_fixed_target_map(weights)
    for name, weights
    in FIXED_PORTFOLIOS.items()
}

regime_target_map = {}

allocation_records = []

for date in rebalance_dates:
    current_regime = test_signals.loc[
        date,
        "regime",
    ]

    target_weights = pd.Series(
        REGIME_WEIGHTS[current_regime],
        index=TICKERS,
        dtype=float,
    )

    regime_target_map[date] = (
        target_weights.copy()
    )

    record = {
        "date": date,
        "signal_source_date": (
            test_signals.loc[
                date,
                "signal_source_date",
            ]
        ),
        "regime": current_regime,
    }

    record.update(
        target_weights.to_dict()
    )

    allocation_records.append(
        record
    )

pd.DataFrame(
    allocation_records
).to_csv(
    ALLOCATIONS_FILE,
    index=False,
)


# ==================================================
# 9. PORTFOLIO SIMULATION
# ==================================================

def simulate_portfolio(
    returns,
    target_weights_by_date,
):
    previous_end_weights = None
    records = []

    for date, monthly_returns in returns.iterrows():

        if date in target_weights_by_date:
            start_weights = (
                target_weights_by_date[
                    date
                ].copy()
            )

            if previous_end_weights is None:
                turnover = 0.0

            else:
                turnover = 0.5 * (
                    start_weights
                    - previous_end_weights
                ).abs().sum()

            rebalanced = True

        else:
            if previous_end_weights is None:
                raise RuntimeError(
                    "The simulation began "
                    "without target weights."
                )

            start_weights = (
                previous_end_weights.copy()
            )

            turnover = 0.0
            rebalanced = False

        gross_return = float(
            np.dot(
                start_weights.to_numpy(),
                monthly_returns.to_numpy(),
            )
        )

        transaction_cost = (
            turnover
            * TRANSACTION_COST_RATE
        )

        net_return = (
            gross_return
            - transaction_cost
        )

        ending_values = (
            start_weights
            * (1.0 + monthly_returns)
        )

        ending_portfolio_value = float(
            ending_values.sum()
        )

        if ending_portfolio_value <= 0:
            raise RuntimeError(
                "Portfolio value became "
                f"non-positive on {date.date()}."
            )

        end_weights = (
            ending_values
            / ending_portfolio_value
        )

        record = {
            "date": date,
            "gross_return": gross_return,
            "transaction_cost": (
                transaction_cost
            ),
            "net_return": net_return,
            "turnover": turnover,
            "rebalanced": rebalanced,
        }

        for ticker in TICKERS:
            record[
                f"start_weight_{ticker}"
            ] = start_weights[ticker]

            record[
                f"end_weight_{ticker}"
            ] = end_weights[ticker]

        records.append(record)

        previous_end_weights = (
            end_weights
        )

    simulation = pd.DataFrame(
        records
    )

    simulation = simulation.set_index(
        "date"
    )

    return simulation


simulations = {}

for portfolio_name, target_map in (
    fixed_target_maps.items()
):
    simulations[portfolio_name] = (
        simulate_portfolio(
            test_returns,
            target_map,
        )
    )

simulations["Regime-Sensitive"] = (
    simulate_portfolio(
        test_returns,
        regime_target_map,
    )
)


# ==================================================
# 10. LOAD OPTIMIZED PORTFOLIO RESULTS
# ==================================================

optimized_results = pd.read_csv(
    OPTIMIZED_FILE,
    parse_dates=["date"],
)

optimized_results = (
    optimized_results
    .set_index("date")
    .sort_index()
)

optimized_return_columns = {
    "Mean-Variance (Max Sharpe)": (
        "net_return_mean_variance"
    ),

    "Equal Risk Contribution": (
        "net_return_equal_risk"
    ),

    "Minimum CVaR": (
        "net_return_minimum_cvar"
    ),
}

optimized_turnover_columns = {
    "Mean-Variance (Max Sharpe)": (
        "turnover_mean_variance"
    ),

    "Equal Risk Contribution": (
        "turnover_equal_risk"
    ),

    "Minimum CVaR": (
        "turnover_minimum_cvar"
    ),
}

required_optimized_columns = (
    list(
        optimized_return_columns.values()
    )
    + list(
        optimized_turnover_columns.values()
    )
)

missing_optimized_columns = [
    column
    for column in required_optimized_columns
    if column not in optimized_results.columns
]

if missing_optimized_columns:
    raise RuntimeError(
        "Optimized result columns are missing:\n"
        f"{missing_optimized_columns}"
    )

if not optimized_results.index.equals(
    test_returns.index
):
    raise RuntimeError(
        "The optimized-results dates do not "
        "exactly match the fair-comparison dates."
    )


# ==================================================
# 11. COMBINE ALL SIX PORTFOLIOS
# ==================================================

net_returns = pd.DataFrame(
    index=test_returns.index
)

turnover_table = pd.DataFrame(
    index=test_returns.index
)

for portfolio_name, simulation in (
    simulations.items()
):
    net_returns[portfolio_name] = (
        simulation["net_return"]
    )

    turnover_table[portfolio_name] = (
        simulation["turnover"]
    )

for portfolio_name, column_name in (
    optimized_return_columns.items()
):
    net_returns[portfolio_name] = (
        optimized_results[column_name]
    )

for portfolio_name, column_name in (
    optimized_turnover_columns.items()
):
    turnover_table[portfolio_name] = (
        optimized_results[column_name]
    )

portfolio_order = [
    "Traditional 60/40",
    "Strategic Diversified",
    "Mean-Variance (Max Sharpe)",
    "Equal Risk Contribution",
    "Minimum CVaR",
    "Regime-Sensitive",
]

net_returns = net_returns[
    portfolio_order
]

turnover_table = turnover_table[
    portfolio_order
]

monthly_output = pd.DataFrame(
    index=test_returns.index
)

for portfolio_name in portfolio_order:
    safe_name = (
        portfolio_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )

    monthly_output[
        f"net_return_{safe_name}"
    ] = net_returns[
        portfolio_name
    ]

    monthly_output[
        f"turnover_{safe_name}"
    ] = turnover_table[
        portfolio_name
    ]

monthly_output["regime"] = (
    test_signals["regime"]
)

monthly_output["signal_source_date"] = (
    test_signals["signal_source_date"]
)

monthly_output.to_csv(
    MONTHLY_OUTPUT_FILE
)


# ==================================================
# 12. PERFORMANCE FUNCTIONS
# ==================================================

def calculate_maximum_drawdown(
    returns,
):
    wealth = (
        1.0 + returns
    ).cumprod()

    drawdown = (
        wealth
        / wealth.cummax()
        - 1.0
    )

    return float(
        drawdown.min()
    )


def calculate_longest_drawdown(
    returns,
):
    wealth = (
        1.0 + returns
    ).cumprod()

    underwater = (
        wealth
        < wealth.cummax()
    )

    longest = 0
    current = 0

    for is_underwater in underwater:
        if is_underwater:
            current += 1

            longest = max(
                longest,
                current,
            )

        else:
            current = 0

    return longest


def calculate_metrics(
    portfolio_returns,
    portfolio_turnover,
):
    portfolio_returns = (
        portfolio_returns.dropna()
    )

    risk_free_returns = (
        test_returns["BIL"]
        .reindex(
            portfolio_returns.index
        )
    )


    months = len(
        portfolio_returns
    )

    years = (
        months
        / MONTHS_PER_YEAR
    )

    nominal_growth = float(
        (
            1.0
            + portfolio_returns
        ).prod()
    )

    real_growth=float(
        nominal_growth
        / CUMULATIVE_INFLATION_FACTOR
        )


    cagr = (
        nominal_growth
        ** (1.0 / years)
        - 1.0
    )

    real_cagr = (
        real_growth
        ** (1.0 / years)
        - 1.0
    )

    annualized_volatility = float(
        portfolio_returns.std(ddof=1)
        * np.sqrt(MONTHS_PER_YEAR)
    )

    excess_returns = (
        portfolio_returns
        - risk_free_returns
    )

    excess_volatility = float(
        excess_returns.std(ddof=1)
    )

    if excess_volatility > 0:
        sharpe_ratio = float(
            excess_returns.mean()
            / excess_volatility
            * np.sqrt(MONTHS_PER_YEAR)
        )

    else:
        sharpe_ratio = np.nan

    downside_returns = np.minimum(
        excess_returns,
        0.0,
    )

    downside_deviation = float(
        np.sqrt(
            np.mean(
                downside_returns ** 2
            )
        )
        * np.sqrt(MONTHS_PER_YEAR)
    )

    if downside_deviation > 0:
        sortino_ratio = float(
            (
                excess_returns.mean()
                * MONTHS_PER_YEAR
            )
            / downside_deviation
        )

    else:
        sortino_ratio = np.nan

    fifth_percentile = float(
        portfolio_returns.quantile(
            0.05
        )
    )

    tail_returns = portfolio_returns[
        portfolio_returns
        <= fifth_percentile
    ]

    return {
        "Months": months,
        "CAGR": cagr,
        "Real CAGR": real_cagr,
        "Annualized Volatility": (
            annualized_volatility
        ),
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": (
            calculate_maximum_drawdown(
                portfolio_returns
            )
        ),
        "Longest Drawdown (Months)": (
            calculate_longest_drawdown(
                portfolio_returns
            )
        ),
        "Best Month": float(
            portfolio_returns.max()
        ),
        "Worst Month": float(
            portfolio_returns.min()
        ),
        "95% VaR": (
            -fifth_percentile
        ),
        "95% CVaR": float(
            -tail_returns.mean()
        ),
        "Average Annual Turnover": float(
            portfolio_turnover.sum()
            / years
        ),
        "Final Wealth of $1": (
            nominal_growth
        ),
    }


# ==================================================
# 13. CALCULATE PERFORMANCE TABLE
# ==================================================

performance_records = []

for portfolio_name in portfolio_order:
    metrics = calculate_metrics(
        net_returns[portfolio_name],
        turnover_table[portfolio_name],
    )

    metrics["Portfolio"] = (
        portfolio_name
    )

    performance_records.append(
        metrics
    )

performance_table = pd.DataFrame(
    performance_records
)

performance_table = (
    performance_table
    .set_index("Portfolio")
)

performance_table.to_csv(
    METRICS_CSV_FILE
)

with pd.ExcelWriter(
    METRICS_EXCEL_FILE,
    engine="openpyxl",
) as writer:

    performance_table.to_excel(
        writer,
        sheet_name="Raw Metrics",
    )

    formatted_table = (
        performance_table.copy()
    )

    percentage_columns = [
        "CAGR",
        "Real CAGR",
        "Annualized Volatility",
        "Maximum Drawdown",
        "Best Month",
        "Worst Month",
        "95% VaR",
        "95% CVaR",
        "Average Annual Turnover",
    ]

    for column in percentage_columns:
        formatted_table[column] = (
            formatted_table[column]
            .map(
                lambda value: (
                    f"{value:.2%}"
                )
            )
        )

    for column in [
        "Sharpe Ratio",
        "Sortino Ratio",
        "Final Wealth of $1",
    ]:
        formatted_table[column] = (
            formatted_table[column]
            .map(
                lambda value: (
                    f"{value:.3f}"
                )
            )
        )

    formatted_table.to_excel(
        writer,
        sheet_name="Formatted Metrics",
    )


# ==================================================
# 14. ANNUAL RETURNS
# ==================================================

annual_returns = (
    1.0 + net_returns
).groupby(
    net_returns.index.year
).prod() - 1.0

annual_returns.index.name = "year"

annual_returns.to_excel(
    ANNUAL_RETURNS_FILE,
    engine="openpyxl",
)


# ==================================================
# 15. PERFORMANCE BY REGIME
# ==================================================

regime_records = []

for regime_name in REGIME_WEIGHTS:

    regime_mask = (
        test_signals["regime"]
        == regime_name
    )

    for portfolio_name in portfolio_order:

        returns_in_regime = (
            net_returns.loc[
                regime_mask,
                portfolio_name,
            ]
        )

        months = len(
            returns_in_regime
        )

        if months == 0:
            continue

        annualized_return = (
            (
                1.0
                + returns_in_regime
            ).prod()
            ** (
                MONTHS_PER_YEAR
                / months
            )
            - 1.0
        )

        if months > 1:
            annualized_volatility = (
                returns_in_regime.std(
                    ddof=1
                )
                * np.sqrt(
                    MONTHS_PER_YEAR
                )
            )

        else:
            annualized_volatility = (
                np.nan
            )

        regime_records.append(
            {
                "Regime": regime_name,
                "Portfolio": portfolio_name,
                "Months": months,
                "Annualized Return": (
                    annualized_return
                ),
                "Annualized Volatility": (
                    annualized_volatility
                ),
                "Average Monthly Return": (
                    returns_in_regime.mean()
                ),
                "Worst Month": (
                    returns_in_regime.min()
                ),
                "Positive-Month Rate": (
                    returns_in_regime
                    .gt(0)
                    .mean()
                ),
            }
        )

regime_table = pd.DataFrame(
    regime_records
)

regime_counts = (
    test_signals["regime"]
    .value_counts()
    .rename_axis("Regime")
    .reset_index(name="Months")
)

with pd.ExcelWriter(
    REGIME_RESULTS_FILE,
    engine="openpyxl",
) as writer:

    regime_table.to_excel(
        writer,
        sheet_name="Performance by Regime",
        index=False,
    )

    regime_counts.to_excel(
        writer,
        sheet_name="Regime Counts",
        index=False,
    )


# ==================================================
# 16. CUMULATIVE WEALTH FIGURE
# ==================================================

cumulative_wealth = (
    1.0 + net_returns
).cumprod()

figure, axis = plt.subplots(
    figsize=(12, 7)
)

for portfolio_name in portfolio_order:
    axis.plot(
        cumulative_wealth.index,
        cumulative_wealth[
            portfolio_name
        ],
        label=portfolio_name,
        linewidth=2,
    )

axis.set_title(
    "Growth of $1: Fair Comparison, 2013–2025"
)

axis.set_xlabel("Date")
axis.set_ylabel("Portfolio Value")
axis.grid(True, alpha=0.3)
axis.legend()

figure.tight_layout()

figure.savefig(
    WEALTH_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


# ==================================================
# 17. DRAWDOWN FIGURE
# ==================================================

drawdowns = pd.DataFrame(
    index=net_returns.index
)

for portfolio_name in portfolio_order:

    portfolio_wealth = (
        1.0
        + net_returns[
            portfolio_name
        ]
    ).cumprod()

    drawdowns[portfolio_name] = (
        portfolio_wealth
        / portfolio_wealth.cummax()
        - 1.0
    )

figure, axis = plt.subplots(
    figsize=(12, 7)
)

for portfolio_name in portfolio_order:
    axis.plot(
        drawdowns.index,
        drawdowns[
            portfolio_name
        ],
        label=portfolio_name,
        linewidth=2,
    )

axis.set_title(
    "Drawdowns: Fair Comparison, 2013–2025"
)

axis.set_xlabel("Date")
axis.set_ylabel("Drawdown")
axis.grid(True, alpha=0.3)
axis.legend()

axis.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda value, position: (
            f"{value:.0%}"
        )
    )
)

figure.tight_layout()

figure.savefig(
    DRAWDOWN_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


# ==================================================
# 18. DISPLAY RESULTS
# ==================================================

display_table = (
    performance_table.copy()
)

for column in [
    "CAGR",
    "Real CAGR",
    "Annualized Volatility",
    "Maximum Drawdown",
    "Best Month",
    "Worst Month",
    "95% VaR",
    "95% CVaR",
    "Average Annual Turnover",
]:
    display_table[column] = (
        display_table[column]
        .map(
            lambda value: (
                f"{value:.2%}"
            )
        )
    )

for column in [
    "Sharpe Ratio",
    "Sortino Ratio",
    "Final Wealth of $1",
]:
    display_table[column] = (
        display_table[column]
        .map(
            lambda value: (
                f"{value:.3f}"
            )
        )
    )

print("\n" + "=" * 72)
print("REGIME-SENSITIVE AND FAIR COMPARISON COMPLETED")
print("=" * 72)

print("\nFair 2013–2025 performance results:")

print(
    display_table.to_string()
)

print(
    "\nMonthly comparison saved to:\n"
    f"{MONTHLY_OUTPUT_FILE}"
)

print(
    "\nPerformance metrics saved to:\n"
    f"{METRICS_EXCEL_FILE}"
)

print(
    "\nRegime signal history saved to:\n"
    f"{SIGNALS_FILE}"
)

print(
    "\nRegime allocations saved to:\n"
    f"{ALLOCATIONS_FILE}"
)

print(
    "\nPerformance by regime saved to:\n"
    f"{REGIME_RESULTS_FILE}"
)

print(
    "\nCumulative wealth figure saved to:\n"
    f"{WEALTH_FIGURE}"
)

print(
    "\nDrawdown figure saved to:\n"
    f"{DRAWDOWN_FIGURE}"
)