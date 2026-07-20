from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT = Path(__file__).resolve().parent.parent

PROCESSED = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    PROCESSED / "validated_monthly_dataset.csv"
)

MONTHLY_FILE = (
    TABLES / "optimized_monthly_results.csv"
)

METRICS_FILE = (
    TABLES / "optimized_performance_metrics.xlsx"
)

WEIGHTS_FILE = (
    TABLES / "optimized_rebalance_weights.xlsx"
)

DIAGNOSTICS_FILE = (
    TABLES / "optimization_diagnostics.csv"
)

WEALTH_FIGURE = (
    FIGURES / "optimized_cumulative_wealth.png"
)

DRAWDOWN_FIGURE = (
    FIGURES / "optimized_drawdowns.png"
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

ANALYSIS_START = pd.Timestamp(
    "2008-01-31"
)

OUT_OF_SAMPLE_START = pd.Timestamp(
    "2013-01-31"
)

LOOKBACK_MONTHS = 60

REBALANCE_MONTHS = {
    1,
    4,
    7,
    10,
}

TRANSACTION_COST = 0.001

CVAR_LEVEL = 0.95

MONTHS_PER_YEAR = 12

INDEX = {
    ticker: index
    for index, ticker in enumerate(TICKERS)
}

N_ASSETS = len(TICKERS)


# ==================================================
# 3. PORTFOLIO CONSTRAINTS
# ==================================================

# BIL has a minimum weight of 5%.
LOWER = np.array(
    [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.05,
    ],
    dtype=float,
)

# No individual asset may exceed 40%.
UPPER = np.array(
    [0.40] * N_ASSETS,
    dtype=float,
)

# Feasible starting point for the optimizers.
INITIAL = np.array(
    [
        0.35,
        0.15,
        0.15,
        0.10,
        0.10,
        0.10,
        0.05,
    ],
    dtype=float,
)

BOUNDS = list(
    zip(
        LOWER,
        UPPER,
    )
)


def slsqp_constraints():
    """
    Return the constraints used by the SLSQP models.
    """

    return [
        {
            "type": "eq",
            "fun": lambda weights: (
                weights.sum() - 1.0
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                0.70
                - weights[INDEX["SPY"]]
                - weights[INDEX["VEA"]]
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                0.30
                - weights[INDEX["VNQ"]]
                - weights[INDEX["GLD"]]
            ),
        },
    ]


def validate_weights(
    weights,
    model,
    date,
    tolerance=1e-5,
):
    """
    Confirm that every optimized portfolio follows
    the predetermined constraints.
    """

    if not np.isfinite(weights).all():
        raise RuntimeError(
            f"{model} produced invalid weights "
            f"on {date.date()}."
        )

    valid_sum = np.isclose(
        weights.sum(),
        1.0,
        atol=tolerance,
    )

    valid_lower_bounds = np.all(
        weights >= LOWER - tolerance
    )

    valid_upper_bounds = np.all(
        weights <= UPPER + tolerance
    )

    equity_weight = (
        weights[INDEX["SPY"]]
        + weights[INDEX["VEA"]]
    )

    real_asset_weight = (
        weights[INDEX["VNQ"]]
        + weights[INDEX["GLD"]]
    )

    valid_equity_cap = (
        equity_weight
        <= 0.70 + tolerance
    )

    valid_real_asset_cap = (
        real_asset_weight
        <= 0.30 + tolerance
    )

    checks = [
        valid_sum,
        valid_lower_bounds,
        valid_upper_bounds,
        valid_equity_cap,
        valid_real_asset_cap,
    ]

    if not all(checks):
        raise RuntimeError(
            f"{model} violated a portfolio constraint "
            f"on {date.date()}."
        )


# ==================================================
# 4. LOAD THE VALIDATED DATA
# ==================================================

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

missing_columns = [
    column
    for column in RETURN_COLUMNS
    if column not in data.columns
]

if missing_columns:
    raise RuntimeError(
        "The following return columns are missing:\n"
        f"{missing_columns}"
    )

all_returns = data[
    RETURN_COLUMNS
].copy()

all_returns.columns = TICKERS

all_returns = all_returns.loc[
    ANALYSIS_START:
].copy()

if all_returns.isna().any().any():
    raise RuntimeError(
        "Missing ETF returns exist after the "
        "analysis start date."
    )

test_returns = all_returns.loc[
    OUT_OF_SAMPLE_START:
].copy()

if test_returns.empty:
    raise RuntimeError(
        "The out-of-sample dataset is empty."
    )

first_test_date = (
    test_returns.index.min()
)

available_training_rows = all_returns.loc[
    all_returns.index < first_test_date
]

if (
    len(available_training_rows)
    < LOOKBACK_MONTHS
):
    raise RuntimeError(
        "There are not enough observations for "
        "the first 60-month estimation window."
    )

print("=" * 68)
print("OPTIMIZED PORTFOLIO ANALYSIS")
print("=" * 68)

print(
    "Estimation data: "
    f"{all_returns.index.min().date()} through "
    f"{all_returns.index.max().date()}"
)

print(
    "Out-of-sample test: "
    f"{test_returns.index.min().date()} through "
    f"{test_returns.index.max().date()}"
)

print(
    f"Out-of-sample months: "
    f"{len(test_returns)}"
)


# ==================================================
# 5. OPTIMIZATION MODELS
# ==================================================

def covariance_matrix(window):
    """
    Calculate the covariance matrix and add a tiny
    diagonal value for numerical stability.
    """

    covariance = (
        window.cov().to_numpy()
    )

    covariance = (
        covariance
        + np.eye(N_ASSETS) * 1e-10
    )

    return covariance


def optimize_mean_variance(window):
    """
    Maximize the historical Sharpe ratio using only
    the previous 60 months of information.
    """

    expected_returns = (
        window.mean().to_numpy()
    )

    covariance = covariance_matrix(
        window
    )

    risk_free_return = float(
        window["BIL"].mean()
    )

    def objective(weights):
        portfolio_return = float(
            weights @ expected_returns
        )

        portfolio_variance = float(
            weights
            @ covariance
            @ weights
        )

        if portfolio_variance <= 0:
            return 1_000_000.0

        portfolio_volatility = np.sqrt(
            portfolio_variance
        )

        sharpe_ratio = (
            portfolio_return
            - risk_free_return
        ) / portfolio_volatility

        # Minimize the negative Sharpe ratio.
        return -float(sharpe_ratio)

    result = minimize(
        objective,
        INITIAL,
        method="SLSQP",
        bounds=BOUNDS,
        constraints=slsqp_constraints(),
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Mean-variance optimization failed:\n"
            f"{result.message}"
        )

    return (
        result.x.astype(float),
        float(result.fun),
        str(result.message),
    )


def optimize_equal_risk(window):
    """
    Make each asset's contribution to total portfolio
    variance as equal as possible.
    """

    covariance = covariance_matrix(
        window
    )

    equal_risk_target = (
        1.0 / N_ASSETS
    )

    def objective(weights):
        portfolio_variance = float(
            weights
            @ covariance
            @ weights
        )

        if portfolio_variance <= 0:
            return 1_000_000.0

        marginal_variance = (
            covariance @ weights
        )

        variance_contributions = (
            weights * marginal_variance
        )

        contribution_shares = (
            variance_contributions
            / portfolio_variance
        )

        squared_errors = (
            contribution_shares
            - equal_risk_target
        ) ** 2

        return float(
            squared_errors.sum()
        )

    result = minimize(
        objective,
        INITIAL,
        method="SLSQP",
        bounds=BOUNDS,
        constraints=slsqp_constraints(),
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Equal-risk optimization failed:\n"
            f"{result.message}"
        )

    return (
        result.x.astype(float),
        float(result.fun),
        str(result.message),
    )


def optimize_minimum_cvar(window):
    """
    Minimize historical monthly 95% CVaR using
    linear programming.
    """

    historical_returns = (
        window.to_numpy()
    )

    number_of_scenarios = len(window)

    # Variables consist of:
    # 7 asset weights
    # 1 VaR threshold
    # 1 loss slack for each historical month

    variable_count = (
        N_ASSETS
        + 1
        + number_of_scenarios
    )

    var_index = N_ASSETS

    slack_start = (
        N_ASSETS + 1
    )

    objective = np.zeros(
        variable_count
    )

    objective[var_index] = 1.0

    objective[
        slack_start:
    ] = (
        1.0
        / (
            (1.0 - CVAR_LEVEL)
            * number_of_scenarios
        )
    )

    inequality_rows = []
    inequality_limits = []

    # Historical loss constraint:
    #
    # loss - VaR - slack <= 0
    #
    # loss equals the negative portfolio return.

    for scenario in range(
        number_of_scenarios
    ):
        row = np.zeros(
            variable_count
        )

        row[:N_ASSETS] = (
            -historical_returns[scenario]
        )

        row[var_index] = -1.0

        row[
            slack_start + scenario
        ] = -1.0

        inequality_rows.append(row)
        inequality_limits.append(0.0)

    # SPY plus VEA cannot exceed 70%.

    equity_row = np.zeros(
        variable_count
    )

    equity_row[
        INDEX["SPY"]
    ] = 1.0

    equity_row[
        INDEX["VEA"]
    ] = 1.0

    inequality_rows.append(
        equity_row
    )

    inequality_limits.append(
        0.70
    )

    # VNQ plus GLD cannot exceed 30%.

    real_asset_row = np.zeros(
        variable_count
    )

    real_asset_row[
        INDEX["VNQ"]
    ] = 1.0

    real_asset_row[
        INDEX["GLD"]
    ] = 1.0

    inequality_rows.append(
        real_asset_row
    )

    inequality_limits.append(
        0.30
    )

    # All portfolio weights must sum to one.

    equality_matrix = np.zeros(
        (1, variable_count)
    )

    equality_matrix[
        0,
        :N_ASSETS,
    ] = 1.0

    equality_limits = np.array(
        [1.0]
    )

    variable_bounds = list(
        zip(
            LOWER,
            UPPER,
        )
    )

    # VaR threshold has no fixed bound.
    variable_bounds.append(
        (None, None)
    )

    # Loss slacks cannot be negative.
    variable_bounds.extend(
        [(0.0, None)]
        * number_of_scenarios
    )

    result = linprog(
        c=objective,
        A_ub=np.asarray(
            inequality_rows
        ),
        b_ub=np.asarray(
            inequality_limits
        ),
        A_eq=equality_matrix,
        b_eq=equality_limits,
        bounds=variable_bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            "Minimum-CVaR optimization failed:\n"
            f"{result.message}"
        )

    optimized_weights = result.x[
        :N_ASSETS
    ].astype(float)

    return (
        optimized_weights,
        float(result.fun),
        str(result.message),
    )


MODELS = {
    "mean_variance": (
        "Mean-Variance (Max Sharpe)",
        optimize_mean_variance,
    ),
    "equal_risk": (
        "Equal Risk Contribution",
        optimize_equal_risk,
    ),
    "minimum_cvar": (
        "Minimum CVaR",
        optimize_minimum_cvar,
    ),
}


# ==================================================
# 6. CALCULATE QUARTERLY TARGET WEIGHTS
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
        "The out-of-sample period must begin in "
        "a scheduled rebalance month."
    )

targets = {
    model_key: {}
    for model_key in MODELS
}

weight_records = []
diagnostic_records = []

print("\nRunning rolling optimizations...")

for rebalance_date in rebalance_dates:

    historical_window = all_returns.loc[
        all_returns.index < rebalance_date
    ].tail(
        LOOKBACK_MONTHS
    )

    if (
        len(historical_window)
        != LOOKBACK_MONTHS
    ):
        raise RuntimeError(
            "The estimation window for "
            f"{rebalance_date.date()} contains "
            f"{len(historical_window)} observations."
        )

    if (
        historical_window.index.max()
        >= rebalance_date
    ):
        raise RuntimeError(
            "Look-ahead bias was detected for "
            f"{rebalance_date.date()}."
        )

    print(
        f"Optimizing "
        f"{rebalance_date.date()}..."
    )

    for model_key, model_information in (
        MODELS.items()
    ):
        display_name = (
            model_information[0]
        )

        optimizer = (
            model_information[1]
        )

        (
            optimized_weights,
            objective_value,
            solver_message,
        ) = optimizer(
            historical_window
        )

        validate_weights(
            optimized_weights,
            display_name,
            rebalance_date,
        )

        weight_series = pd.Series(
            optimized_weights,
            index=TICKERS,
            dtype=float,
        )

        targets[
            model_key
        ][rebalance_date] = (
            weight_series
        )

        weight_record = {
            "date": rebalance_date,
            "model_key": model_key,
            "Portfolio": display_name,
            "window_start": (
                historical_window.index.min()
            ),
            "window_end": (
                historical_window.index.max()
            ),
        }

        weight_record.update(
            weight_series.to_dict()
        )

        weight_records.append(
            weight_record
        )

        diagnostic_records.append(
            {
                "date": rebalance_date,
                "model_key": model_key,
                "Portfolio": display_name,
                "window_start": (
                    historical_window.index.min()
                ),
                "window_end": (
                    historical_window.index.max()
                ),
                "observations": len(
                    historical_window
                ),
                "objective_value": (
                    objective_value
                ),
                "solver_message": (
                    solver_message
                ),
            }
        )


# ==================================================
# 7. SIMULATE THE PORTFOLIOS
# ==================================================

def simulate(
    returns,
    target_weights,
):
    """
    Simulate quarterly rebalancing and weight drift.
    """

    previous_end_weights = None

    records = []

    for date, asset_return in returns.iterrows():

        if date in target_weights:
            start_weights = (
                target_weights[date].copy()
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
                    "The portfolio simulation began "
                    "without target weights."
                )

            start_weights = (
                previous_end_weights.copy()
            )

            turnover = 0.0
            rebalanced = False

        gross_return = float(
            start_weights
            @ asset_return
        )

        transaction_cost = float(
            turnover
            * TRANSACTION_COST
        )

        net_return = (
            gross_return
            - transaction_cost
        )

        ending_values = (
            start_weights
            * (1.0 + asset_return)
        )

        ending_portfolio_value = float(
            ending_values.sum()
        )

        if ending_portfolio_value <= 0:
            raise RuntimeError(
                "Portfolio value became non-positive "
                f"on {date.date()}."
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
            "turnover": float(turnover),
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

for model_key, model_information in (
    MODELS.items()
):
    display_name = (
        model_information[0]
    )

    print(
        f"\nSimulating {display_name}..."
    )

    simulations[model_key] = simulate(
        test_returns,
        targets[model_key],
    )


# ==================================================
# 8. PERFORMANCE FUNCTIONS
# ==================================================

def maximum_drawdown(returns):
    wealth = (
        1.0 + returns
    ).cumprod()

    running_peak = wealth.cummax()

    drawdown = (
        wealth
        / running_peak
        - 1.0
    )

    return float(
        drawdown.min()
    )


def longest_drawdown(returns):
    wealth = (
        1.0 + returns
    ).cumprod()

    underwater = (
        wealth
        < wealth.cummax()
    )

    longest = 0
    current = 0

    for value in underwater:
        if value:
            current += 1
        else:
            current = 0

        longest = max(
            longest,
            current,
        )

    return longest


def calculate_metrics(
    portfolio_returns,
    risk_free_returns,
    turnover,
):
    portfolio_returns = (
        portfolio_returns.dropna()
    )

    risk_free_returns = (
        risk_free_returns.reindex(
            portfolio_returns.index
        )
    )

    if risk_free_returns.isna().any():
        raise RuntimeError(
            "Risk-free returns are missing during "
            "the test period."
        )

    months = len(
        portfolio_returns
    )

    years = (
        months
        / MONTHS_PER_YEAR
    )

    total_growth = float(
        (
            1.0
            + portfolio_returns
        ).prod()
    )

    cagr = (
        total_growth ** (1.0 / years)
        - 1.0
    )

    volatility = float(
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
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": (
            maximum_drawdown(
                portfolio_returns
            )
        ),
        "Longest Drawdown (Months)": (
            longest_drawdown(
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
            turnover.sum()
            / years
        ),
        "Final Wealth of $1": (
            total_growth
        ),
    }


# ==================================================
# 9. SAVE OUTPUT TABLES
# ==================================================

monthly_results = pd.DataFrame(
    index=test_returns.index
)

monthly_results.index.name = "date"

net_returns = pd.DataFrame(
    index=test_returns.index
)

net_returns.index.name = "date"

performance_records = []

for model_key, model_information in (
    MODELS.items()
):
    display_name = (
        model_information[0]
    )

    simulation = (
        simulations[model_key]
    )

    monthly_results[
        f"gross_return_{model_key}"
    ] = simulation["gross_return"]

    monthly_results[
        f"net_return_{model_key}"
    ] = simulation["net_return"]

    monthly_results[
        f"turnover_{model_key}"
    ] = simulation["turnover"]

    monthly_results[
        f"transaction_cost_{model_key}"
    ] = simulation[
        "transaction_cost"
    ]

    monthly_results[
        f"rebalanced_{model_key}"
    ] = simulation["rebalanced"]

    net_returns[
        display_name
    ] = simulation["net_return"]

    performance = calculate_metrics(
        simulation["net_return"],
        test_returns["BIL"],
        simulation["turnover"],
    )

    performance[
        "Portfolio"
    ] = display_name

    performance_records.append(
        performance
    )

monthly_results.to_csv(
    MONTHLY_FILE
)

performance_table = pd.DataFrame(
    performance_records
)

performance_table = (
    performance_table.set_index(
        "Portfolio"
    )
)

with pd.ExcelWriter(
    METRICS_FILE,
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
            formatted_table[column].map(
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
            formatted_table[column].map(
                lambda value: (
                    f"{value:.3f}"
                )
            )
        )

    formatted_table.to_excel(
        writer,
        sheet_name="Formatted Metrics",
    )


weight_table = pd.DataFrame(
    weight_records
)

weight_table = weight_table.sort_values(
    [
        "model_key",
        "date",
    ]
)
WEIGHTS_CSV_FILE = (
    TABLES / "optimized_rebalance_weights.csv"
)

weight_table.to_csv(
    WEIGHTS_CSV_FILE,
    index=False,
)

with pd.ExcelWriter(
    WEIGHTS_FILE,
    engine="openpyxl",
) as writer:

    weight_table.to_excel(
        writer,
        sheet_name="All Models",
        index=False,
    )

    for model_key in MODELS:

        model_weights = weight_table.loc[
            weight_table["model_key"]
            == model_key
        ]

        model_weights.to_excel(
            writer,
            sheet_name=model_key[:31],
            index=False,
        )


diagnostics_table = pd.DataFrame(
    diagnostic_records
)

diagnostics_table = (
    diagnostics_table.sort_values(
        [
            "date",
            "model_key",
        ]
    )
)

diagnostics_table.to_csv(
    DIAGNOSTICS_FILE,
    index=False,
)


# ==================================================
# 10. CREATE FIGURES
# ==================================================

cumulative_wealth = (
    1.0 + net_returns
).cumprod()

figure, axis = plt.subplots(
    figsize=(11, 6)
)

for portfolio_name in (
    cumulative_wealth.columns
):
    axis.plot(
        cumulative_wealth.index,
        cumulative_wealth[
            portfolio_name
        ],
        label=portfolio_name,
        linewidth=2,
    )

axis.set_title(
    "Growth of $1: Optimized Portfolios"
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


drawdowns = pd.DataFrame(
    index=net_returns.index
)

for portfolio_name in net_returns.columns:

    portfolio_wealth = (
        1.0
        + net_returns[
            portfolio_name
        ]
    ).cumprod()

    drawdowns[
        portfolio_name
    ] = (
        portfolio_wealth
        / portfolio_wealth.cummax()
        - 1.0
    )

figure, axis = plt.subplots(
    figsize=(11, 6)
)

for portfolio_name in drawdowns.columns:
    axis.plot(
        drawdowns.index,
        drawdowns[
            portfolio_name
        ],
        label=portfolio_name,
        linewidth=2,
    )

axis.set_title(
    "Drawdowns: Optimized Portfolios"
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
# 11. DISPLAY RESULTS
# ==================================================

display_table = (
    performance_table.copy()
)

for column in [
    "CAGR",
    "Annualized Volatility",
    "Maximum Drawdown",
    "Best Month",
    "Worst Month",
    "95% VaR",
    "95% CVaR",
    "Average Annual Turnover",
]:
    display_table[column] = (
        display_table[column].map(
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
        display_table[column].map(
            lambda value: (
                f"{value:.3f}"
            )
        )
    )

print("\n" + "=" * 68)
print("OPTIMIZED PORTFOLIO ANALYSIS COMPLETED")
print("=" * 68)

print("\nOut-of-sample results:")
print(
    display_table.to_string()
)

print(
    "\nMonthly results saved to:\n"
    f"{MONTHLY_FILE}"
)

print(
    "\nPerformance metrics saved to:\n"
    f"{METRICS_FILE}"
)

print(
    "\nRebalance weights saved to:\n"
    f"{WEIGHTS_FILE}"
)
print(
    "\nCSV rebalance weights saved to:\n"
    f"{WEIGHTS_CSV_FILE}"
)

print(
    "\nOptimization diagnostics saved to:\n"
    f"{DIAGNOSTICS_FILE}"
)

print(
    "\nCumulative wealth figure saved to:\n"
    f"{WEALTH_FIGURE}"
)

print(
    "\nDrawdown figure saved to:\n"
    f"{DRAWDOWN_FIGURE}"
)