"""Monte Carlo market-risk analysis using Yahoo Finance price history.
 --ticker SPY --horizon 252 --simulations 20000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import yfinance as yf


TRADING_DAYS = 252


@dataclass(frozen=True)
class RiskReport:
    ticker: str
    spot_price: float
    expected_price: float
    expected_return: float
    volatility: float
    var_95: float
    cvar_95: float
    probability_of_loss: float
    worst_case: float
    best_case: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical Monte Carlo risk analysis")
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance symbol, e.g. SPY or AAPL")
    parser.add_argument(
        "--portfolio",
        help="Portfolio weights formatted as TICKER:WEIGHT,..., e.g. SPY:0.6,AAPL:0.4",
    )
    parser.add_argument("--lookback", type=int, default=5, help="Years of historical prices")
    parser.add_argument("--horizon", type=int, default=252, help="Trading days to simulate")
    parser.add_argument("--simulations", type=int, default=20_000, help="Number of Monte Carlo paths")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for VaR/CVaR")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results")
    parser.add_argument("--output", default="monte_carlo_risk.png", help="Output chart filename")
    return parser.parse_args()


def parse_portfolio(value: str) -> dict[str, float]:
    portfolio: dict[str, float] = {}
    for position in value.split(","):
        try:
            ticker, weight = position.split(":")
            portfolio[ticker.strip().upper()] = float(weight)
        except ValueError as error:
            raise ValueError(f"Invalid portfolio position '{position}'. Use TICKER:WEIGHT.") from error

    if not portfolio or any(weight < 0 for weight in portfolio.values()):
        raise ValueError("Portfolio weights must be non-negative.")
    total_weight = sum(portfolio.values())
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"Portfolio weights must sum to 1.0; received {total_weight:.4f}.")
    return portfolio


def download_prices(ticker: str, lookback_years: int) -> pd.Series:
    if lookback_years < 1:
        raise ValueError("lookback must be at least 1 year")

    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = end - pd.DateOffset(years=lookback_years)
    prices = yf.download(
        ticker.upper(),
        start=start.date(),
        end=end.date(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if prices.empty:
        raise ValueError(f"No price history found for '{ticker}'. Check the ticker symbol.")

    close = prices["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    if len(close) < 60:
        raise ValueError(f"Only {len(close)} price observations found; at least 60 are required.")
    return close


def download_portfolio_prices(tickers: list[str], lookback_years: int) -> pd.DataFrame:
    histories = {ticker: download_prices(ticker, lookback_years) for ticker in tickers}
    prices = pd.concat(histories, axis=1).dropna()
    if len(prices) < 60:
        raise ValueError("The portfolio has fewer than 60 shared trading days of history.")
    return prices


def simulate_paths(
    spot_price: float,
    log_returns: pd.Series,
    horizon: int,
    simulations: int,
    seed: int,
) -> np.ndarray:
    if horizon < 1 or simulations < 1:
        raise ValueError("horizon and simulations must be positive")

    daily_drift = float(log_returns.mean())
    daily_volatility = float(log_returns.std(ddof=1))
    if daily_volatility <= 0:
        raise ValueError("Historical volatility is zero; simulation cannot proceed.")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((horizon, simulations))
    daily_returns = daily_drift + daily_volatility * shocks
    cumulative_returns = np.exp(np.cumsum(daily_returns, axis=0))
    growth_factors = np.exp(np.cumsum(daily_returns, axis=0))
    starting_row = np.full((1, simulations), spot_price)
    price_paths = spot_price * growth_factors
    return np.vstack([starting_row, price_paths])


def simulate_portfolio_paths(
    prices: pd.DataFrame,
    weights: np.ndarray,
    horizon: int,
    simulations: int,
    seed: int,
) -> tuple[np.ndarray, pd.Series]:
    if horizon < 1 or simulations < 1:
        raise ValueError("horizon and simulations must be positive")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_drift = log_returns.mean().to_numpy()
    covariance = log_returns.cov().to_numpy()
    portfolio_start = float(np.dot(prices.iloc[-1].to_numpy(), weights))
    rng = np.random.default_rng(seed)
    shocks = rng.multivariate_normal(
        mean=np.zeros(len(weights)),
        cov=covariance,
        size=(horizon, simulations),
    )
    daily_returns = daily_drift + shocks
    asset_growth = np.exp(np.cumsum(daily_returns, axis=0))
    portfolio_growth = np.einsum("dsa,a->ds", asset_growth, weights)
    starting_row = np.full((1, simulations), portfolio_start)
    portfolio_paths = portfolio_start * portfolio_growth
    simulated_log_returns = pd.Series(np.log(prices / prices.shift(1)).dot(weights))
    return np.vstack([starting_row, portfolio_paths]), simulated_log_returns

def build_report(
    ticker: str,
    paths: np.ndarray,
    log_returns: pd.Series,
    confidence: float,
) -> RiskReport:
    terminal_prices = paths[-1]
    spot_price = float(paths[0, 0])
    terminal_returns = terminal_prices / spot_price - 1
    loss_returns = np.sort(terminal_returns)
    tail_start = int(np.floor((1 - confidence) * len(loss_returns)))
    var_return = float(loss_returns[tail_start])
    cvar_return = float(loss_returns[: max(1, tail_start)].mean())
    return RiskReport(
        ticker=ticker.upper(),
        spot_price=spot_price,
        expected_price=float(terminal_prices.mean()),
        expected_return=float(terminal_returns.mean()),
        volatility=float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        var_95=-var_return,
        cvar_95=-cvar_return,
        probability_of_loss=float((terminal_returns < 0).mean()),
        worst_case=float(terminal_returns.min()),
        best_case=float(terminal_returns.max()),
    )


def plot_results(
    ticker: str,
    paths: np.ndarray,
    report: RiskReport,
    output_path: str,
) -> None:
    days = np.arange(paths.shape[0])
    percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    terminal_prices = paths[-1]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.65, 1]},
    )
    figure.patch.set_facecolor("#f7f4ee")
    for axis in axes:
        axis.set_facecolor("#f7f4ee")

    sample_count = min(150, paths.shape[1])
    axes[0].plot(days, paths[:, :sample_count], color="#8aa6a3", alpha=0.10, linewidth=0.8)
    axes[0].fill_between(days, percentiles[0], percentiles[4], color="#2f6f73", alpha=0.16, label="5th-95th percentile")
    axes[0].plot(days, percentiles[2], color="#c85c3d", linewidth=2.4, label="Median path")
    axes[0].axhline(report.spot_price, color="#263238", linewidth=1, linestyle="--", label="Starting price")
    axes[0].set_title(f"{ticker.upper()} simulated price paths", loc="left", weight="bold")
    axes[0].set_xlabel("Trading days")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].hist(terminal_prices, bins=60, color="#2f6f73", alpha=0.88, edgecolor="#f7f4ee")
    axes[1].axvline(report.spot_price, color="#263238", linestyle="--", linewidth=1.5, label="Today")
    axes[1].axvline(report.spot_price * (1 - report.var_95), color="#c85c3d", linewidth=2, label="95% VaR threshold")
    axes[1].set_title("Terminal price distribution", loc="left", weight="bold")
    axes[1].set_xlabel("Price ($)")
    axes[1].set_ylabel("Simulated outcomes")
    axes[1].legend(frameon=False, loc="upper left")

    figure.suptitle(
        f"CLASSICAL MONTE CARLO RISK LAB  |  {ticker.upper()}  |  {paths.shape[1]:,} paths",
        x=0.06,
        ha="left",
        fontsize=15,
        weight="bold",
        color="#263238",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def print_report(report: RiskReport, horizon: int, simulations: int, confidence: float, output_path: str) -> None:
    print("\nCLASSICAL MONTE CARLO RISK REPORT")
    print("=" * 38)
    print(f"Asset                 {report.ticker}")
    print(f"Horizon               {horizon} trading days")
    print(f"Simulations           {simulations:,}")
    print(f"Starting price        ${report.spot_price:,.2f}")
    print(f"Annualized volatility {report.volatility:.2%}")
    print(f"Expected terminal     ${report.expected_price:,.2f} ({report.expected_return:+.2%})")
    print(f"Probability of loss   {report.probability_of_loss:.2%}")
    print(f"VaR ({confidence:.0%})            {report.var_95:.2%}")
    print(f"CVaR ({confidence:.0%})           {report.cvar_95:.2%}")
    print(f"Simulated range       {report.worst_case:+.2%} to {report.best_case:+.2%}")
    print(f"Chart saved to        {Path(output_path).resolve()}\n")


def main() -> None:
    args = parse_args()
    if not 0.5 < args.confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")
    if args.portfolio:
        portfolio = parse_portfolio(args.portfolio)
        prices = download_portfolio_prices(list(portfolio), args.lookback)
        weights = np.array([portfolio[ticker] for ticker in prices.columns])
        paths, log_returns = simulate_portfolio_paths(
            prices, weights, args.horizon, args.simulations, args.seed
        )
        label = "Portfolio (" + ", ".join(portfolio) + ")"
    else:
        prices = download_prices(args.ticker, args.lookback)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        paths = simulate_paths(prices.iloc[-1], log_returns, args.horizon, args.simulations, args.seed)
        label = args.ticker

    report = build_report(label, paths, log_returns, args.confidence)
    plot_results(label, paths, report, args.output)
    print_report(report, args.horizon, args.simulations, args.confidence, args.output)


if __name__ == "__main__":
    main()