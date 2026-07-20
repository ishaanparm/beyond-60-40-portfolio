from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
PROCESSED_FOLDER = PROJECT_FOLDER / "data" / "processed"
TABLES_FOLDER = PROJECT_FOLDER / "outputs" / "tables"
TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

DATA_FILE = PROCESSED_FOLDER / "validated_monthly_dataset.csv"
FAIR_FILE = TABLES_FOLDER / "fair_comparison_monthly_results.csv"

COST_OUTPUT = TABLES_FOLDER / "robustness_transaction_costs.xlsx"
STRESS_OUTPUT = TABLES_FOLDER / "robustness_stress_periods.xlsx"
VARIANT_OUTPUT = TABLES_FOLDER / "robustness_mean_variance_variants.xlsx"
SUMMARY_OUTPUT = TABLES_FOLDER / "robustness_summary.txt"


# ==================================================
# 2. SETTINGS
# ==================================================

TICKERS = ["SPY", "VEA", "IEF", "LQD", "VNQ", "GLD", "BIL"]
RETURN_COLUMNS = [f"return_{ticker}" for ticker in TICKERS]

MAIN_START = pd.Timestamp("2013-01-31")
ROBUSTNESS_START = pd.Timestamp("2015-01-31")
MONTHS_PER_YEAR = 12
BASE_COST_RATE = 0.001

PORTFOLIO_COLUMNS = {
    "Traditional 60/40": (
        "net_return_traditional_60_40",
        "turnover_traditional_60_40",
    ),
    "Strategic Diversified": (
        "net_return_strategic_diversified",
        "turnover_strategic_diversified",
    ),
    "Mean-Variance (Max Sharpe)": (
        "net_return_mean_variance_max_sharpe",
        "turnover_mean_variance_max_sharpe",
    ),
    "Equal Risk Contribution": (
        "net_return_equal_risk_contribution",
        "turnover_equal_risk_contribution",
    ),
    "Minimum CVaR": (
        "net_return_minimum_cvar",
        "turnover_minimum_cvar",
    ),
    "Regime-Sensitive": (
        "net_return_regime_sensitive",
        "turnover_regime_sensitive",
    ),
}

COST_SCENARIOS = {
    "0 bps": 0.0000,
    "10 bps": 0.0010,
    "25 bps": 0.0025,
}

STRESS_PERIODS = {
    "COVID Shock": ("2020-02-29", "2020-03-31"),
    "COVID Full Year": ("2020-01-31", "2020-12-31"),
    "2022 Rate Shock": ("2022-01-31", "2022-12-31"),
}

REBALANCE_MONTHS = {
    "Monthly": set(range(1, 13)),
    "Quarterly": {1, 4, 7, 10},
    "Semiannual": {1, 7},
}


# ==================================================
# 3. LOAD DATA
# ==================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "validated_monthly_dataset.csv was not found. "
        "Run 02_data_validation.py first."
    )

if not FAIR_FILE.exists():
    raise FileNotFoundError(
        "fair_comparison_monthly_results.csv was not found. "
        "Run 05_regime_sensitive_comparison.py first."
    )

data = pd.read_csv(DATA_FILE, parse_dates=["date"]).set_index("date")
data = data.sort_index()

fair = pd.read_csv(FAIR_FILE, parse_dates=["date"]).set_index("date")
fair = fair.sort_index()

required_data = RETURN_COLUMNS + ["nber_recession"]
missing_data = [column for column in required_data if column not in data.columns]
if missing_data:
    raise RuntimeError(f"Missing validated-data columns: {missing_data}")

required_fair = [
    column
    for pair in PORTFOLIO_COLUMNS.values()
    for column in pair
]
missing_fair = [column for column in required_fair if column not in fair.columns]
if missing_fair:
    raise RuntimeError(f"Missing fair-comparison columns: {missing_fair}")

main_index = data.loc[MAIN_START:].index
if not fair.index.equals(main_index):
    raise RuntimeError(
        "The fair-comparison dates do not match the 2013-2025 test period."
    )

asset_returns = data[RETURN_COLUMNS].copy()
asset_returns.columns = TICKERS
asset_returns = asset_returns.loc[pd.Timestamp("2008-01-31"):]

if asset_returns.isna().any().any():
    raise RuntimeError("Missing ETF returns exist after January 2008.")

print("=" * 72)
print("ROBUSTNESS AND STRESS-TEST ANALYSIS")
print("=" * 72)
print(
    f"Main period: {fair.index.min().date()} through "
    f"{fair.index.max().date()}"
)
print(
    f"Variant period: {ROBUSTNESS_START.date()} through "
    f"{asset_returns.index.max().date()}"
)


# ==================================================
# 4. METRIC FUNCTIONS
# ==================================================

def maximum_drawdown(returns):
    """Include the initial wealth value of 1.0."""
    wealth = np.concatenate(
        ([1.0], (1.0 + returns).cumprod().to_numpy())
    )
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peak - 1.0
    return float(drawdowns.min())


def performance_metrics(returns, risk_free, turnover):
    returns = returns.dropna()
    risk_free = risk_free.reindex(returns.index)
    turnover = turnover.reindex(returns.index).fillna(0.0)

    if risk_free.isna().any():
        raise RuntimeError("Risk-free returns are missing in a test.")

    months = len(returns)
    years = months / MONTHS_PER_YEAR
    growth = float((1.0 + returns).prod())
    cagr = growth ** (1.0 / years) - 1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))

    excess = returns - risk_free
    excess_volatility = float(excess.std(ddof=1))
    sharpe = (
        float(excess.mean() / excess_volatility * np.sqrt(MONTHS_PER_YEAR))
        if excess_volatility > 0
        else np.nan
    )

    fifth_percentile = float(returns.quantile(0.05))
    tail = returns[returns <= fifth_percentile]

    return {
        "Months": months,
        "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": maximum_drawdown(returns),
        "Worst Month": float(returns.min()),
        "95% CVaR": float(-tail.mean()),
        "Average Annual Turnover": float(turnover.sum() / years),
        "Final Wealth of $1": growth,
    }


def stress_metrics(returns):
    returns = returns.dropna()
    if returns.empty:
        return {
            "Months": 0,
            "Cumulative Return": np.nan,
            "Worst Month": np.nan,
            "Maximum Drawdown": np.nan,
        }

    return {
        "Months": len(returns),
        "Cumulative Return": float((1.0 + returns).prod() - 1.0),
        "Worst Month": float(returns.min()),
        "Maximum Drawdown": maximum_drawdown(returns),
    }


# ==================================================
# 5. TRANSACTION-COST SENSITIVITY
# ==================================================

cost_records = []
risk_free_main = asset_returns.loc[fair.index, "BIL"]

for portfolio_name, (return_column, turnover_column) in PORTFOLIO_COLUMNS.items():
    baseline_net = fair[return_column]
    turnover = fair[turnover_column]

    # File 05 used 10 bps, so reconstruct gross returns first.
    gross_returns = baseline_net + turnover * BASE_COST_RATE

    for scenario_name, cost_rate in COST_SCENARIOS.items():
        scenario_returns = gross_returns - turnover * cost_rate
        metrics = performance_metrics(
            scenario_returns,
            risk_free_main,
            turnover,
        )
        metrics["Portfolio"] = portfolio_name
        metrics["Transaction Cost"] = scenario_name
        cost_records.append(metrics)

cost_table = pd.DataFrame(cost_records).set_index(
    ["Portfolio", "Transaction Cost"]
)

with pd.ExcelWriter(COST_OUTPUT, engine="openpyxl") as writer:
    cost_table.to_excel(writer, sheet_name="Raw Results")

    formatted = cost_table.copy()
    for column in [
        "CAGR",
        "Annualized Volatility",
        "Maximum Drawdown",
        "Worst Month",
        "95% CVaR",
        "Average Annual Turnover",
    ]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")

    for column in ["Sharpe Ratio", "Final Wealth of $1"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")

    formatted.to_excel(writer, sheet_name="Formatted Results")


# ==================================================
# 6. STRESS TESTS
# ==================================================

stress_records = []

for stress_name, (start_date, end_date) in STRESS_PERIODS.items():
    for portfolio_name, (return_column, _) in PORTFOLIO_COLUMNS.items():
        period_returns = fair.loc[start_date:end_date, return_column]
        metrics = stress_metrics(period_returns)
        metrics.update(
            {
                "Stress Period": stress_name,
                "Portfolio": portfolio_name,
                "Start Date": pd.Timestamp(start_date),
                "End Date": pd.Timestamp(end_date),
            }
        )
        stress_records.append(metrics)

recession_indicator = data.loc[fair.index, "nber_recession"]
recession_mask = recession_indicator.eq(1)

for portfolio_name, (return_column, _) in PORTFOLIO_COLUMNS.items():
    recession_returns = fair.loc[recession_mask, return_column]
    metrics = stress_metrics(recession_returns)
    metrics.update(
        {
            "Stress Period": "NBER Recession Months",
            "Portfolio": portfolio_name,
            "Start Date": (
                recession_returns.index.min()
                if not recession_returns.empty
                else pd.NaT
            ),
            "End Date": (
                recession_returns.index.max()
                if not recession_returns.empty
                else pd.NaT
            ),
        }
    )
    stress_records.append(metrics)

stress_table = pd.DataFrame(stress_records)[
    [
        "Stress Period",
        "Portfolio",
        "Start Date",
        "End Date",
        "Months",
        "Cumulative Return",
        "Worst Month",
        "Maximum Drawdown",
    ]
]

with pd.ExcelWriter(STRESS_OUTPUT, engine="openpyxl") as writer:
    stress_table.to_excel(writer, sheet_name="All Stress Tests", index=False)

    for stress_name in stress_table["Stress Period"].unique():
        stress_table.loc[
            stress_table["Stress Period"].eq(stress_name)
        ].to_excel(
            writer,
            sheet_name=stress_name[:31],
            index=False,
        )


# ==================================================
# 7. MEAN-VARIANCE VARIANT ENGINE
# ==================================================

ASSET_INDEX = {ticker: position for position, ticker in enumerate(TICKERS)}
N_ASSETS = len(TICKERS)

LOWER_WEIGHTS = np.array(
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05],
    dtype=float,
)

INITIAL_WEIGHTS = np.array(
    [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10],
    dtype=float,
)


def optimize_mean_variance(window, maximum_weight):
    """
    Maximize the historical Sharpe ratio.

    Multiple feasible starting portfolios are tested
    because SLSQP can occasionally become stuck.
    """

    expected_returns = (
        window.mean().to_numpy()
    )

    covariance = (
        window.cov().to_numpy()
        + np.eye(N_ASSETS) * 1e-10
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

        sharpe = (
            portfolio_return
            - risk_free_return
        ) / portfolio_volatility

        return -float(sharpe)

    constraints = [
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
                - weights[ASSET_INDEX["SPY"]]
                - weights[ASSET_INDEX["VEA"]]
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                0.30
                - weights[ASSET_INDEX["VNQ"]]
                - weights[ASSET_INDEX["GLD"]]
            ),
        },
    ]

    upper_weights = np.array(
        [maximum_weight] * N_ASSETS,
        dtype=float,
    )

    bounds = list(
        zip(
            LOWER_WEIGHTS,
            upper_weights,
        )
    )

    # Different feasible starting portfolios.
    starting_points = [
        np.array(
            [
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                0.10,
            ],
            dtype=float,
        ),
        np.array(
            [
                1 / 7,
                1 / 7,
                1 / 7,
                1 / 7,
                1 / 7,
                1 / 7,
                1 / 7,
            ],
            dtype=float,
        ),
        np.array(
            [
                0.20,
                0.05,
                0.25,
                0.15,
                0.05,
                0.05,
                0.25,
            ],
            dtype=float,
        ),
        np.array(
            [
                0.30,
                0.10,
                0.20,
                0.10,
                0.10,
                0.05,
                0.15,
            ],
            dtype=float,
        ),
    ]

    def is_feasible(weights):
        tolerance = 1e-5

        if not np.isfinite(weights).all():
            return False

        if not np.isclose(
            weights.sum(),
            1.0,
            atol=tolerance,
        ):
            return False

        if np.any(
            weights
            < LOWER_WEIGHTS - tolerance
        ):
            return False

        if np.any(
            weights
            > upper_weights + tolerance
        ):
            return False

        equity_weight = (
            weights[ASSET_INDEX["SPY"]]
            + weights[ASSET_INDEX["VEA"]]
        )

        real_asset_weight = (
            weights[ASSET_INDEX["VNQ"]]
            + weights[ASSET_INDEX["GLD"]]
        )

        if equity_weight > 0.70 + tolerance:
            return False

        if real_asset_weight > 0.30 + tolerance:
            return False

        return True

    successful_results = []
    failure_messages = []

    for starting_point in starting_points:
        if not is_feasible(starting_point):
            continue

        result = minimize(
            objective,
            starting_point,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 5000,
                "ftol": 1e-9,
                "disp": False,
            },
        )

        if (
            result.success
            and is_feasible(result.x)
            and np.isfinite(result.fun)
        ):
            successful_results.append(
                result
            )
        else:
            failure_messages.append(
                str(result.message)
            )

    if not successful_results:
        messages = "; ".join(
            sorted(set(failure_messages))
        )

        raise RuntimeError(
            "Mean-variance robustness optimization "
            "failed for every starting portfolio. "
            f"Solver messages: {messages}"
        )

    # Select the feasible result with the highest
    # Sharpe ratio, meaning the lowest objective.
    best_result = min(
        successful_results,
        key=lambda result: result.fun,
    )

    weights = pd.Series(
        best_result.x,
        index=TICKERS,
        dtype=float,
    )

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            "A robustness portfolio does not sum "
            "to one."
        )

    return weights


def run_variant(lookback, frequency, maximum_weight, cost_rate):
    test_returns = asset_returns.loc[ROBUSTNESS_START:].copy()
    allowed_months = REBALANCE_MONTHS[frequency]
    rebalance_dates = [
        date for date in test_returns.index if date.month in allowed_months
    ]

    if test_returns.index.min() not in rebalance_dates:
        raise RuntimeError("The robustness period must begin on a rebalance date.")

    target_weights = {}
    weight_records = []

    for rebalance_date in rebalance_dates:
        window = asset_returns.loc[
            asset_returns.index < rebalance_date
        ].tail(lookback)

        if len(window) != lookback:
            raise RuntimeError(
                f"Only {len(window)} rows were available for the "
                f"{lookback}-month window on {rebalance_date.date()}."
            )

        weights = optimize_mean_variance(window, maximum_weight)
        target_weights[rebalance_date] = weights

        record = {
            "date": rebalance_date,
            "window_start": window.index.min(),
            "window_end": window.index.max(),
        }
        record.update(weights.to_dict())
        weight_records.append(record)

    previous_end_weights = None
    records = []

    for date, monthly_returns in test_returns.iterrows():
        if date in target_weights:
            start_weights = target_weights[date].copy()
            turnover = (
                0.0
                if previous_end_weights is None
                else 0.5 * (start_weights - previous_end_weights).abs().sum()
            )
        else:
            if previous_end_weights is None:
                raise RuntimeError("The variant began without target weights.")
            start_weights = previous_end_weights.copy()
            turnover = 0.0

        gross_return = float(start_weights @ monthly_returns)
        net_return = gross_return - turnover * cost_rate

        ending_values = start_weights * (1.0 + monthly_returns)
        end_weights = ending_values / ending_values.sum()

        records.append(
            {
                "date": date,
                "net_return": net_return,
                "turnover": float(turnover),
            }
        )
        previous_end_weights = end_weights

    results = pd.DataFrame(records).set_index("date")
    weights_table = pd.DataFrame(weight_records)

    return results, weights_table


# ==================================================
# 8. RUN VARIANTS
# ==================================================

VARIANTS = [
    {
        "Variant": "Lookback 36M",
        "Lookback": 36,
        "Frequency": "Quarterly",
        "Maximum Weight": 0.40,
        "Cost": 0.001,
    },
    {
        "Variant": "Main 60M",
        "Lookback": 60,
        "Frequency": "Quarterly",
        "Maximum Weight": 0.40,
        "Cost": 0.001,
    },
    {
        "Variant": "Lookback 84M",
        "Lookback": 84,
        "Frequency": "Quarterly",
        "Maximum Weight": 0.40,
        "Cost": 0.001,
    },
    {
        "Variant": "Monthly Rebalance",
        "Lookback": 60,
        "Frequency": "Monthly",
        "Maximum Weight": 0.40,
        "Cost": 0.001,
    },
    {
        "Variant": "Semiannual Rebalance",
        "Lookback": 60,
        "Frequency": "Semiannual",
        "Maximum Weight": 0.40,
        "Cost": 0.001,
    },
    {
        "Variant": "30% Asset Cap",
        "Lookback": 60,
        "Frequency": "Quarterly",
        "Maximum Weight": 0.30,
        "Cost": 0.001,
    },
]

variant_records = []
variant_weight_tables = []
risk_free_variants = asset_returns.loc[ROBUSTNESS_START:, "BIL"]

print("\nRunning mean-variance variants...")

for variant in VARIANTS:
    print(f"Running {variant['Variant']}...")

    results, weights_table = run_variant(
        lookback=variant["Lookback"],
        frequency=variant["Frequency"],
        maximum_weight=variant["Maximum Weight"],
        cost_rate=variant["Cost"],
    )

    metrics = performance_metrics(
        results["net_return"],
        risk_free_variants,
        results["turnover"],
    )
    metrics.update(variant)
    variant_records.append(metrics)

    weights_table.insert(0, "Variant", variant["Variant"])
    variant_weight_tables.append(weights_table)

variant_table = pd.DataFrame(variant_records)[
    [
        "Variant",
        "Lookback",
        "Frequency",
        "Maximum Weight",
        "Cost",
        "Months",
        "CAGR",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Worst Month",
        "95% CVaR",
        "Average Annual Turnover",
        "Final Wealth of $1",
    ]
]

all_variant_weights = pd.concat(variant_weight_tables, ignore_index=True)

with pd.ExcelWriter(VARIANT_OUTPUT, engine="openpyxl") as writer:
    variant_table.to_excel(writer, sheet_name="Variant Metrics", index=False)
    all_variant_weights.to_excel(writer, sheet_name="Variant Weights", index=False)


# ==================================================
# 9. SUMMARY
# ==================================================

main_cost_results = cost_table.xs("10 bps", level="Transaction Cost")

best_growth = main_cost_results["CAGR"].idxmax()
best_sharpe = main_cost_results["Sharpe Ratio"].idxmax()
best_drawdown = main_cost_results["Maximum Drawdown"].idxmax()
best_tail = main_cost_results["95% CVaR"].idxmin()

best_variant = variant_table.loc[
    variant_table["Sharpe Ratio"].idxmax(),
    "Variant",
]

summary_text = f"""
Beyond the 60/40 Portfolio
Robustness and Stress-Test Summary

MAIN 2013-2025 RESULTS AT 10 BPS
Highest CAGR: {best_growth}
Highest Sharpe ratio: {best_sharpe}
Smallest maximum drawdown: {best_drawdown}
Lowest 95% CVaR: {best_tail}

MEAN-VARIANCE ROBUSTNESS, 2015-2025
Best Sharpe variant: {best_variant}
Lowest variant CAGR: {variant_table['CAGR'].min():.2%}
Highest variant CAGR: {variant_table['CAGR'].max():.2%}

INTERPRETATION
A finding is considered robust only when its direction remains broadly
consistent across transaction-cost, lookback-window, rebalance-frequency,
and weight-cap variations. Historical robustness does not guarantee future
performance.
""".strip()

SUMMARY_OUTPUT.write_text(summary_text, encoding="utf-8")


# ==================================================
# 10. DISPLAY RESULTS
# ==================================================

display_variants = variant_table.copy()

for column in [
    "Maximum Weight",
    "Cost",
    "CAGR",
    "Annualized Volatility",
    "Maximum Drawdown",
    "Worst Month",
    "95% CVaR",
    "Average Annual Turnover",
]:
    display_variants[column] = display_variants[column].map(
        lambda value: f"{value:.2%}"
    )

for column in ["Sharpe Ratio", "Final Wealth of $1"]:
    display_variants[column] = display_variants[column].map(
        lambda value: f"{value:.3f}"
    )

print("\n" + "=" * 72)
print("ROBUSTNESS AND STRESS TESTS COMPLETED")
print("=" * 72)
print("\nMean-variance variants:")
print(display_variants.to_string(index=False))
print(f"\nTransaction-cost results:\n{COST_OUTPUT}")
print(f"\nStress-test results:\n{STRESS_OUTPUT}")
print(f"\nMean-variance variants:\n{VARIANT_OUTPUT}")
print(f"\nSummary:\n{SUMMARY_OUTPUT}")
