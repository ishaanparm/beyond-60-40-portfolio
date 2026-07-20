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

INPUT_FILE = PROCESSED_FOLDER / "validated_monthly_dataset.csv"

MONTHLY_RESULTS_FILE = (
    TABLES_FOLDER / "baseline_monthly_results.csv"
)

PERFORMANCE_CSV_FILE = (
    TABLES_FOLDER / "baseline_performance_metrics.csv"
)

PERFORMANCE_EXCEL_FILE = (
    TABLES_FOLDER / "baseline_performance_metrics.xlsx"
)

ANNUAL_RETURNS_FILE = (
    TABLES_FOLDER / "baseline_annual_returns.xlsx"
)

ALLOCATIONS_FILE = (
    TABLES_FOLDER / "baseline_target_allocations.xlsx"
)

CUMULATIVE_WEALTH_FIGURE = (
    FIGURES_FOLDER / "baseline_cumulative_wealth.png"
)

DRAWDOWN_FIGURE = (
    FIGURES_FOLDER / "baseline_drawdowns.png"
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

# Rebalance before the returns for January, April,
# July, and October are earned.
REBALANCE_MONTHS = {1, 4, 7, 10}

# Ten basis points of one-way turnover.
TRANSACTION_COST_RATE = 0.001

MONTHS_PER_YEAR = 12

ANALYSIS_START_DATE = "2008-01-31"

# ==================================================
# 3. PORTFOLIO DEFINITIONS
# ==================================================

PORTFOLIOS = {
    "60_40": {
        "display_name": "Traditional 60/40",
        "weights": {
            "SPY": 0.60,
            "VEA": 0.00,
            "IEF": 0.40,
            "LQD": 0.00,
            "VNQ": 0.00,
            "GLD": 0.00,
            "BIL": 0.00,
        },
    },
    "strategic": {
        "display_name": "Strategic Diversified",
        "weights": {
            "SPY": 0.35,
            "VEA": 0.15,
            "IEF": 0.15,
            "LQD": 0.10,
            "VNQ": 0.10,
            "GLD": 0.10,
            "BIL": 0.05,
        },
    },
}


# ==================================================
# 4. LOAD AND CHECK DATA
# ==================================================

print("=" * 65)
print("LOADING VALIDATED MONTHLY DATA")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "The validated dataset was not found.\n"
        "Run 02_data_validation.py first."
    )

data = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"],
)

data = data.set_index("date")
data = data.sort_index()

missing_return_columns = [
    column
    for column in RETURN_COLUMNS
    if column not in data.columns
]

if missing_return_columns:
    raise RuntimeError(
        "The following return columns are missing:\n"
        f"{missing_return_columns}"
    )

asset_returns = data[RETURN_COLUMNS].copy()

asset_returns.columns = TICKERS

asset_returns = asset_returns.loc[
    pd.Timestamp(ANALYSIS_START_DATE):
].copy()

if asset_returns.isna().any().any():
    missing_dates = asset_returns.index[
        asset_returns.isna().any(axis=1)
    ].tolist()

    raise RuntimeError(
        "Missing ETF returns were found in the validated dataset:\n"
        f"{missing_dates}"
    )

print(f"Rows loaded: {len(asset_returns):,}")
print(
    "Sample period: "
    f"{asset_returns.index.min().date()} through "
    f"{asset_returns.index.max().date()}"
)


# ==================================================
# 5. VALIDATE TARGET WEIGHTS
# ==================================================

def validate_weights(
    portfolio_key: str,
    weights: dict[str, float],
) -> None:
    """
    Check that portfolio weights are usable.
    """

    missing_tickers = [
        ticker
        for ticker in TICKERS
        if ticker not in weights
    ]

    extra_tickers = [
        ticker
        for ticker in weights
        if ticker not in TICKERS
    ]

    if missing_tickers:
        raise RuntimeError(
            f"{portfolio_key} is missing weights for:\n"
            f"{missing_tickers}"
        )

    if extra_tickers:
        raise RuntimeError(
            f"{portfolio_key} contains unknown tickers:\n"
            f"{extra_tickers}"
        )

    weight_series = pd.Series(
        weights,
        index=TICKERS,
        dtype=float,
    )

    if (weight_series < 0).any():
        raise RuntimeError(
            f"{portfolio_key} contains a negative weight."
        )

    if not np.isclose(weight_series.sum(), 1.0):
        raise RuntimeError(
            f"{portfolio_key} weights sum to "
            f"{weight_series.sum():.6f}, not 1.0."
        )


for portfolio_key, portfolio_info in PORTFOLIOS.items():
    validate_weights(
        portfolio_key,
        portfolio_info["weights"],
    )

print("\nAll target portfolio weights are valid.")


# ==================================================
# 6. PORTFOLIO SIMULATION
# ==================================================

def simulate_portfolio(
    returns: pd.DataFrame,
    target_weights: dict[str, float],
) -> pd.DataFrame:
    """
    Simulate a quarterly rebalanced portfolio.

    The first allocation is not charged a transaction cost.
    Later rebalances incur a cost based on one-way turnover.
    """

    target = pd.Series(
        target_weights,
        index=TICKERS,
        dtype=float,
    )

    previous_end_weights = None
    records = []

    for date, monthly_asset_returns in returns.iterrows():
        first_month = previous_end_weights is None

        rebalance_month = (
            date.month in REBALANCE_MONTHS
        )

        if first_month:
            start_weights = target.copy()
            turnover = 0.0
            rebalanced = True

        elif rebalance_month:
            turnover = 0.5 * (
                target - previous_end_weights
            ).abs().sum()

            start_weights = target.copy()
            rebalanced = True

        else:
            start_weights = previous_end_weights.copy()
            turnover = 0.0
            rebalanced = False

        gross_return = float(
            np.dot(
                start_weights.to_numpy(),
                monthly_asset_returns.to_numpy(),
            )
        )

        transaction_cost = (
            turnover * TRANSACTION_COST_RATE
        )

        net_return = (
            gross_return - transaction_cost
        )

        ending_values = (
            start_weights
            * (1.0 + monthly_asset_returns)
        )

        ending_portfolio_value = ending_values.sum()

        if ending_portfolio_value <= 0:
            raise RuntimeError(
                "The simulated portfolio value became "
                f"non-positive on {date.date()}."
            )

        end_weights = (
            ending_values
            / ending_portfolio_value
        )

        weight_sum = end_weights.sum()

        if not np.isclose(weight_sum, 1.0):
            raise RuntimeError(
                "Ending portfolio weights do not sum "
                f"to one on {date.date()}."
            )

        record = {
            "date": date,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "turnover": turnover,
            "rebalanced": rebalanced,
        }

        for ticker in TICKERS:
            record[f"start_weight_{ticker}"] = (
                start_weights[ticker]
            )

            record[f"end_weight_{ticker}"] = (
                end_weights[ticker]
            )

        records.append(record)

        previous_end_weights = end_weights

    simulation = pd.DataFrame(records)

    simulation = simulation.set_index("date")

    return simulation


# ==================================================
# 7. RUN PORTFOLIO SIMULATIONS
# ==================================================

simulations = {}

for portfolio_key, portfolio_info in PORTFOLIOS.items():
    print(
        "\nSimulating "
        f"{portfolio_info['display_name']}..."
    )

    simulations[portfolio_key] = simulate_portfolio(
        returns=asset_returns,
        target_weights=portfolio_info["weights"],
    )

print("\nBoth baseline portfolios were simulated.")


# ==================================================
# 8. PERFORMANCE FUNCTIONS
# ==================================================

def calculate_max_drawdown(
    returns: pd.Series,
) -> float:
    """
    Calculate the largest peak-to-trough decline.
    """

    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0

    return float(drawdown.min())


def calculate_longest_drawdown(
    returns: pd.Series,
) -> int:
    """
    Count the longest number of consecutive months
    below a previous portfolio peak.
    """

    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    underwater = wealth < running_peak

    longest_period = 0
    current_period = 0

    for is_underwater in underwater:
        if is_underwater:
            current_period += 1
            longest_period = max(
                longest_period,
                current_period,
            )
        else:
            current_period = 0

    return longest_period


def calculate_performance_metrics(
    portfolio_returns: pd.Series,
    risk_free_returns: pd.Series,
    turnover: pd.Series,
) -> dict[str, float]:
    """
    Calculate portfolio performance statistics.
    """

    portfolio_returns = portfolio_returns.dropna()

    risk_free_returns = risk_free_returns.reindex(
        portfolio_returns.index
    )

    if risk_free_returns.isna().any():
        raise RuntimeError(
            "Risk-free returns are missing during "
            "the performance period."
        )

    months = len(portfolio_returns)
    years = months / MONTHS_PER_YEAR

    total_growth = (
        1.0 + portfolio_returns
    ).prod()

    cagr = (
        total_growth ** (1.0 / years)
        - 1.0
    )

    annualized_volatility = (
        portfolio_returns.std(ddof=1)
        * np.sqrt(MONTHS_PER_YEAR)
    )

    excess_returns = (
        portfolio_returns - risk_free_returns
    )

    excess_standard_deviation = (
        excess_returns.std(ddof=1)
    )

    if excess_standard_deviation > 0:
        sharpe_ratio = (
            excess_returns.mean()
            / excess_standard_deviation
            * np.sqrt(MONTHS_PER_YEAR)
        )
    else:
        sharpe_ratio = np.nan

    downside_excess_returns = np.minimum(
        excess_returns,
        0.0,
    )

    annualized_downside_deviation = (
        np.sqrt(
            np.mean(
                np.square(
                    downside_excess_returns
                )
            )
        )
        * np.sqrt(MONTHS_PER_YEAR)
    )

    annualized_mean_excess_return = (
        excess_returns.mean()
        * MONTHS_PER_YEAR
    )

    if annualized_downside_deviation > 0:
        sortino_ratio = (
            annualized_mean_excess_return
            / annualized_downside_deviation
        )
    else:
        sortino_ratio = np.nan

    fifth_percentile = (
        portfolio_returns.quantile(0.05)
    )

    value_at_risk_95 = -fifth_percentile

    tail_returns = portfolio_returns[
        portfolio_returns <= fifth_percentile
    ]

    conditional_value_at_risk_95 = (
        -tail_returns.mean()
    )

    maximum_drawdown = calculate_max_drawdown(
        portfolio_returns
    )

    longest_drawdown_months = (
        calculate_longest_drawdown(
            portfolio_returns
        )
    )

    average_annual_turnover = (
        turnover.sum() / years
    )

    final_wealth = total_growth

    return {
        "Months": months,
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": maximum_drawdown,
        "Longest Drawdown (Months)": (
            longest_drawdown_months
        ),
        "Best Month": portfolio_returns.max(),
        "Worst Month": portfolio_returns.min(),
        "95% VaR": value_at_risk_95,
        "95% CVaR": conditional_value_at_risk_95,
        "Average Annual Turnover": (
            average_annual_turnover
        ),
        "Final Wealth of $1": final_wealth,
    }


# ==================================================
# 9. BUILD MONTHLY RESULTS TABLE
# ==================================================

monthly_results = pd.DataFrame(
    index=asset_returns.index
)

monthly_results.index.name = "date"

net_return_table = pd.DataFrame(
    index=asset_returns.index
)

net_return_table.index.name = "date"

for portfolio_key, portfolio_info in PORTFOLIOS.items():
    simulation = simulations[portfolio_key]

    monthly_results[
        f"gross_return_{portfolio_key}"
    ] = simulation["gross_return"]

    monthly_results[
        f"net_return_{portfolio_key}"
    ] = simulation["net_return"]

    monthly_results[
        f"turnover_{portfolio_key}"
    ] = simulation["turnover"]

    monthly_results[
        f"transaction_cost_{portfolio_key}"
    ] = simulation["transaction_cost"]

    monthly_results[
        f"rebalanced_{portfolio_key}"
    ] = simulation["rebalanced"]

    net_return_table[
        portfolio_info["display_name"]
    ] = simulation["net_return"]

monthly_results.to_csv(MONTHLY_RESULTS_FILE)


# ==================================================
# 10. CALCULATE PERFORMANCE TABLE
# ==================================================

risk_free_returns = asset_returns["BIL"]

performance_records = []

for portfolio_key, portfolio_info in PORTFOLIOS.items():
    simulation = simulations[portfolio_key]

    metrics = calculate_performance_metrics(
        portfolio_returns=simulation["net_return"],
        risk_free_returns=risk_free_returns,
        turnover=simulation["turnover"],
    )

    metrics["Portfolio"] = (
        portfolio_info["display_name"]
    )

    performance_records.append(metrics)

performance_table = pd.DataFrame(
    performance_records
)

performance_table = performance_table.set_index(
    "Portfolio"
)

performance_table.to_csv(PERFORMANCE_CSV_FILE)

with pd.ExcelWriter(
    PERFORMANCE_EXCEL_FILE,
    engine="openpyxl",
) as writer:
    performance_table.to_excel(
        writer,
        sheet_name="Raw Metrics",
    )

    formatted_performance = performance_table.copy()

    percentage_columns = [
        "CAGR",
        "Annualized Volatility",
        "Maximum Drawdown",
        "Best Month",
        "Worst Month",
        "95% VaR",
        "95% CVaR",
        "Average Annual Turnover",
    ]

    for column in percentage_columns:
        formatted_performance[column] = (
            formatted_performance[column]
            .map(lambda value: f"{value:.2%}")
        )

    ratio_columns = [
        "Sharpe Ratio",
        "Sortino Ratio",
        "Final Wealth of $1",
    ]

    for column in ratio_columns:
        formatted_performance[column] = (
            formatted_performance[column]
            .map(lambda value: f"{value:.3f}")
        )

    formatted_performance.to_excel(
        writer,
        sheet_name="Formatted Metrics",
    )


# ==================================================
# 11. CREATE TARGET ALLOCATION TABLE
# ==================================================

allocation_records = []

for portfolio_key, portfolio_info in PORTFOLIOS.items():
    record = {
        "Portfolio": portfolio_info["display_name"],
    }

    for ticker in TICKERS:
        record[ticker] = (
            portfolio_info["weights"][ticker]
        )

    allocation_records.append(record)

allocation_table = pd.DataFrame(
    allocation_records
)

allocation_table = allocation_table.set_index(
    "Portfolio"
)

allocation_table.to_excel(
    ALLOCATIONS_FILE,
    engine="openpyxl",
)


# ==================================================
# 12. CREATE ANNUAL RETURNS TABLE
# ==================================================

annual_returns = (
    1.0 + net_return_table
).groupby(
    net_return_table.index.year
).prod() - 1.0

annual_returns.index.name = "year"

annual_returns.to_excel(
    ANNUAL_RETURNS_FILE,
    engine="openpyxl",
)


# ==================================================
# 13. CREATE CUMULATIVE WEALTH FIGURE
# ==================================================

cumulative_wealth = (
    1.0 + net_return_table
).cumprod()

figure, axis = plt.subplots(
    figsize=(11, 6)
)

for portfolio_name in cumulative_wealth.columns:
    axis.plot(
        cumulative_wealth.index,
        cumulative_wealth[portfolio_name],
        label=portfolio_name,
        linewidth=2,
    )

axis.set_title(
    "Growth of $1: Baseline Portfolios"
)

axis.set_xlabel("Date")
axis.set_ylabel("Portfolio Value")
axis.grid(True, alpha=0.3)
axis.legend()
figure.tight_layout()

figure.savefig(
    CUMULATIVE_WEALTH_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


# ==================================================
# 14. CREATE DRAWDOWN FIGURE
# ==================================================

drawdown_table = pd.DataFrame(
    index=net_return_table.index
)

for portfolio_name in net_return_table.columns:
    wealth = (
        1.0 + net_return_table[portfolio_name]
    ).cumprod()

    running_peak = wealth.cummax()

    drawdown_table[portfolio_name] = (
        wealth / running_peak - 1.0
    )

figure, axis = plt.subplots(
    figsize=(11, 6)
)

for portfolio_name in drawdown_table.columns:
    axis.plot(
        drawdown_table.index,
        drawdown_table[portfolio_name],
        label=portfolio_name,
        linewidth=2,
    )

axis.set_title(
    "Portfolio Drawdowns"
)

axis.set_xlabel("Date")
axis.set_ylabel("Drawdown")
axis.grid(True, alpha=0.3)
axis.legend()
axis.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda value, position: f"{value:.0%}"
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
# 15. DISPLAY RESULTS
# ==================================================

display_table = performance_table.copy()

display_percentage_columns = [
    "CAGR",
    "Annualized Volatility",
    "Maximum Drawdown",
    "Best Month",
    "Worst Month",
    "95% VaR",
    "95% CVaR",
    "Average Annual Turnover",
]

for column in display_percentage_columns:
    display_table[column] = (
        display_table[column]
        .map(lambda value: f"{value:.2%}")
    )

for column in [
    "Sharpe Ratio",
    "Sortino Ratio",
    "Final Wealth of $1",
]:
    display_table[column] = (
        display_table[column]
        .map(lambda value: f"{value:.3f}")
    )


print("\n" + "=" * 65)
print("BASELINE PORTFOLIO ANALYSIS COMPLETED")
print("=" * 65)

print("\nPerformance results:")
print(display_table.to_string())

print(
    "\nMonthly results saved to:\n"
    f"{MONTHLY_RESULTS_FILE}"
)

print(
    "\nPerformance table saved to:\n"
    f"{PERFORMANCE_EXCEL_FILE}"
)

print(
    "\nAnnual returns saved to:\n"
    f"{ANNUAL_RETURNS_FILE}"
)

print(
    "\nTarget allocations saved to:\n"
    f"{ALLOCATIONS_FILE}"
)

print(
    "\nCumulative wealth figure saved to:\n"
    f"{CUMULATIVE_WEALTH_FIGURE}"
)

print(
    "\nDrawdown figure saved to:\n"
    f"{DRAWDOWN_FIGURE}"
)