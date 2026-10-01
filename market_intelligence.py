"""Product 2: market-wide intelligence tools for S&P and Nasdaq analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf

import monteCarloRisk as risk_model


SP500_CORE = [
    "SPY",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "BRK-B",
    "JPM",
    "XOM",
    "UNH",
]
NASDAQ_CORE = [
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AVGO",
    "COST",
]


@dataclass(frozen=True)
class DiscrepancyFinding:
    severity: str
    category: str
    message: str
    symbol: str | None = None
    details: dict[str, float | str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class SignalCandidate:
    symbol: str
    side: str
    confidence: float
    entry_price: float
    stop_price: float
    target_price: float
    risk_context: str
    thesis: str
    rationale: tuple[str, ...]


@dataclass
class MarketDataset:
    prices: pd.DataFrame
    volumes: pd.DataFrame
    returns: pd.DataFrame
    fundamentals: pd.DataFrame
    symbol_exchange: dict[str, str]
    symbol_sector: dict[str, str]


class MarketDataLayer:
    """Shared data layer for Product 2 market analytics."""

    def __init__(
        self,
        sp500_symbols: list[str] | None = None,
        nasdaq_symbols: list[str] | None = None,
    ) -> None:
        self.sp500_symbols = sp500_symbols or SP500_CORE
        self.nasdaq_symbols = nasdaq_symbols or NASDAQ_CORE

    @property
    def universe(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for symbol in [*self.sp500_symbols, *self.nasdaq_symbols]:
            if symbol not in seen:
                seen.add(symbol)
                ordered.append(symbol)
        return ordered

    def fetch_market_dataset(self, lookback_years: int = 3) -> MarketDataset:
        if lookback_years < 1:
            raise ValueError("lookback_years must be at least 1")
        tickers = self.universe
        end = pd.Timestamp.now(tz="UTC").tz_localize(None)
        start = end - pd.DateOffset(years=lookback_years)
        frame = yf.download(
            tickers,
            start=start.date(),
            end=end.date(),
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        prices = self._extract_field(frame, tickers, "Close")
        volumes = self._extract_field(frame, tickers, "Volume")
        prices = prices.dropna(how="all").ffill()
        volumes = volumes.dropna(how="all").fillna(0.0)
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        fundamentals, sectors = self.fetch_fundamentals(tickers)
        exchange_map = {s: ("S&P 500" if s in self.sp500_symbols else "Nasdaq") for s in tickers}
        return MarketDataset(
            prices=prices,
            volumes=volumes,
            returns=returns,
            fundamentals=fundamentals,
            symbol_exchange=exchange_map,
            symbol_sector=sectors,
        )

    def _extract_field(self, frame: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
        if isinstance(frame.columns, pd.MultiIndex):
            candidates = {}
            for symbol in tickers:
                if (symbol, field) in frame.columns:
                    candidates[symbol] = frame[(symbol, field)]
            if candidates:
                return pd.DataFrame(candidates)
        if field in frame.columns:
            return pd.DataFrame({tickers[0]: frame[field]})
        raise ValueError(f"Unable to extract '{field}' data for requested universe.")

    def fetch_fundamentals(self, tickers: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
        rows = []
        sectors: dict[str, str] = {}
        for symbol in tickers:
            info = yf.Ticker(symbol).info
            sector = str(info.get("sector") or "Unknown")
            sectors[symbol] = sector
            rows.append(
                {
                    "symbol": symbol,
                    "pe_ratio": _safe_float(info.get("trailingPE")),
                    "forward_pe": _safe_float(info.get("forwardPE")),
                    "revenue_growth": _safe_float(info.get("revenueGrowth")),
                    "earnings_growth": _safe_float(info.get("earningsGrowth")),
                    "roe": _safe_float(info.get("returnOnEquity")),
                    "debt_to_equity": _safe_float(info.get("debtToEquity")),
                    "profit_margin": _safe_float(info.get("profitMargins")),
                    "market_cap": _safe_float(info.get("marketCap")),
                }
            )
        return pd.DataFrame(rows).set_index("symbol"), sectors


class DiscrepancyAgent:
    """Asynchronous discrepancy scanner for Product 2 pipelines."""

    def __init__(self) -> None:
        self._findings: list[DiscrepancyFinding] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self, dataset: MarketDataset, on_complete: Callable[[], None] | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            args=(dataset, on_complete),
            daemon=True,
        )
        self._thread.start()

    def _run(self, dataset: MarketDataset, on_complete: Callable[[], None] | None) -> None:
        findings: list[DiscrepancyFinding] = []
        latest_time = dataset.prices.index.max()
        if isinstance(latest_time, pd.Timestamp):
            days_stale = (pd.Timestamp.now() - latest_time).days
            if days_stale > 5:
                findings.append(
                    DiscrepancyFinding(
                        severity="high",
                        category="stale-data",
                        message=f"Latest market data is stale by {days_stale} days.",
                        details={"days_stale": float(days_stale)},
                    )
                )
        missing_rates = dataset.prices.isna().mean().sort_values(ascending=False)
        for symbol, miss_rate in missing_rates.items():
            if miss_rate > 0.05:
                findings.append(
                    DiscrepancyFinding(
                        severity="medium",
                        category="missing-data",
                        symbol=symbol,
                        message=f"{symbol} is missing {miss_rate:.1%} of expected price observations.",
                        details={"missing_rate": float(miss_rate)},
                    )
                )
        extreme_moves = (dataset.returns.abs() > 0.25).sum()
        for symbol, breaches in extreme_moves.items():
            if int(breaches) > 0:
                findings.append(
                    DiscrepancyFinding(
                        severity="medium",
                        category="inconsistent-metrics",
                        symbol=symbol,
                        message=f"{symbol} has {int(breaches)} return observations above 25% in one day.",
                        details={"breaches": float(breaches)},
                    )
                )
        invalid_metric_symbols = dataset.fundamentals.index[dataset.fundamentals["market_cap"].fillna(0) <= 0]
        for symbol in invalid_metric_symbols:
            findings.append(
                DiscrepancyFinding(
                    severity="low",
                    category="failed-calculation",
                    symbol=str(symbol),
                    message=f"{symbol} has incomplete fundamental coverage (non-positive market cap).",
                )
            )
        with self._lock:
            self._findings = findings
            self._running = False
        if on_complete:
            on_complete()

    def snapshot(self) -> list[DiscrepancyFinding]:
        with self._lock:
            return list(self._findings)


class MarketVisualizer:
    """Builds Product 2 visualization-ready analytics tables."""

    def build_regime_health(self, dataset: MarketDataset) -> pd.DataFrame:
        recent = dataset.returns.tail(21)
        annual_vol = dataset.returns.std(ddof=1) * np.sqrt(risk_model.TRADING_DAYS)
        momentum_1m = recent.mean() * 21
        breadth = (recent.tail(1).iloc[0] > 0).astype(float)
        drawdown = dataset.prices / dataset.prices.cummax() - 1
        current_drawdown = drawdown.iloc[-1]
        return pd.DataFrame(
            {
                "momentum_1m": momentum_1m,
                "annual_vol": annual_vol,
                "breadth_flag": breadth,
                "current_drawdown": current_drawdown,
            }
        ).sort_values("momentum_1m", ascending=False)

    def build_heat_map(self, dataset: MarketDataset) -> pd.DataFrame:
        momentum_5d = dataset.prices.pct_change(5).iloc[-1]
        momentum_21d = dataset.prices.pct_change(21).iloc[-1]
        heat = pd.DataFrame(
            {
                "symbol": dataset.prices.columns,
                "exchange": [dataset.symbol_exchange[s] for s in dataset.prices.columns],
                "sector": [dataset.symbol_sector.get(s, "Unknown") for s in dataset.prices.columns],
                "momentum_5d": momentum_5d.values,
                "momentum_21d": momentum_21d.values,
            }
        ).set_index("symbol")
        return heat.sort_values("momentum_21d", ascending=False)

    def build_volatility_liquidity(self, dataset: MarketDataset) -> pd.DataFrame:
        vol = dataset.returns.std(ddof=1) * np.sqrt(risk_model.TRADING_DAYS)
        dollar_volume = (dataset.prices * dataset.volumes).tail(21).mean()
        concentration = (dollar_volume / dollar_volume.sum()).sort_values(ascending=False)
        return pd.DataFrame(
            {
                "annual_vol": vol,
                "avg_dollar_volume_21d": dollar_volume,
                "liquidity_share": concentration,
            }
        ).sort_values("liquidity_share", ascending=False)

    def build_trend_breadth(self, dataset: MarketDataset) -> pd.DataFrame:
        sma_20 = dataset.prices.rolling(20).mean().iloc[-1]
        sma_50 = dataset.prices.rolling(50).mean().iloc[-1]
        latest = dataset.prices.iloc[-1]
        above_20 = latest > sma_20
        above_50 = latest > sma_50
        trend_score = above_20.astype(int) + above_50.astype(int)
        return pd.DataFrame(
            {
                "price": latest,
                "sma20": sma_20,
                "sma50": sma_50,
                "above20": above_20,
                "above50": above_50,
                "trend_score": trend_score,
            }
        ).sort_values("trend_score", ascending=False)


class RiskSignalEngine:
    """Combines risk, heat-map state, and fundamentals to generate long/short candidates."""

    def __init__(self, confidence: float = 0.95, horizon: int = 21, simulations: int = 3000) -> None:
        self.confidence = confidence
        self.horizon = horizon
        self.simulations = simulations

    def build_macro_risk_context(self, dataset: MarketDataset) -> dict[str, risk_model.RiskReport]:
        context: dict[str, risk_model.RiskReport] = {}
        for symbol in ["SPY", "QQQ"]:
            if symbol not in dataset.prices.columns:
                continue
            prices = dataset.prices[symbol].dropna()
            if len(prices) < 60:
                continue
            log_returns = np.log(prices / prices.shift(1)).dropna()
            paths = risk_model.simulate_paths(
                spot_price=float(prices.iloc[-1]),
                log_returns=log_returns,
                horizon=self.horizon,
                simulations=self.simulations,
                seed=42,
            )
            context[symbol] = risk_model.build_report(
                symbol,
                paths,
                log_returns,
                self.confidence,
            )
        return context

    def generate_signals(
        self,
        dataset: MarketDataset,
        heat_map: pd.DataFrame,
        trend_breadth: pd.DataFrame,
        macro_risk: dict[str, risk_model.RiskReport],
        top_n: int = 8,
    ) -> list[SignalCandidate]:
        fundamentals = dataset.fundamentals.reindex(heat_map.index)
        score_frame = pd.DataFrame(index=heat_map.index)
        score_frame["momentum"] = heat_map["momentum_21d"].fillna(0.0)
        score_frame["trend"] = trend_breadth["trend_score"].reindex(heat_map.index).fillna(0.0) / 2
        score_frame["quality"] = (
            fundamentals["roe"].fillna(0.0)
            + fundamentals["profit_margin"].fillna(0.0)
            + fundamentals["revenue_growth"].fillna(0.0)
        )
        score_frame["leverage_penalty"] = fundamentals["debt_to_equity"].fillna(0.0) / 100
        z = score_frame.apply(_zscore).fillna(0.0)
        composite = z["momentum"] * 0.4 + z["trend"] * 0.2 + z["quality"] * 0.35 - z["leverage_penalty"] * 0.25
        regime_bias = self._regime_bias(macro_risk)
        adjusted = composite + regime_bias
        latest_prices = dataset.prices.iloc[-1]
        vol = dataset.returns.std(ddof=1).reindex(adjusted.index).fillna(0.01)
        signals: list[SignalCandidate] = []
        ranked = adjusted.sort_values(ascending=False)
        selected = pd.concat([ranked.head(top_n // 2), ranked.tail(top_n // 2)])
        for symbol, score in selected.items():
            price = float(latest_prices[symbol])
            daily_vol = float(max(vol[symbol], 0.005))
            side = "LONG" if score >= 0 else "SHORT"
            move = 2.5 * daily_vol
            stop = price * (1 - move) if side == "LONG" else price * (1 + move)
            target = price * (1 + move * 1.8) if side == "LONG" else price * (1 - move * 1.8)
            confidence = float(np.clip(0.5 + abs(score) / 4, 0.5, 0.95))
            risk_context = self._build_risk_context_text(macro_risk)
            thesis = "Trend and quality factors align." if side == "LONG" else "Weak trend and quality support downside bias."
            rationale = (
                f"Composite score {score:+.2f}",
                f"21D momentum {heat_map.at[symbol, 'momentum_21d']:+.2%}",
                f"Trend score {trend_breadth.at[symbol, 'trend_score']:.0f}/2",
            )
            signals.append(
                SignalCandidate(
                    symbol=symbol,
                    side=side,
                    confidence=confidence,
                    entry_price=price,
                    stop_price=float(stop),
                    target_price=float(target),
                    risk_context=risk_context,
                    thesis=thesis,
                    rationale=rationale,
                )
            )
        return sorted(signals, key=lambda s: s.confidence, reverse=True)

    def _regime_bias(self, macro_risk: dict[str, risk_model.RiskReport]) -> float:
        if not macro_risk:
            return 0.0
        avg_var = float(np.mean([r.var_95 for r in macro_risk.values()]))
        avg_loss_prob = float(np.mean([r.probability_of_loss for r in macro_risk.values()]))
        return -0.20 if avg_var > 0.08 or avg_loss_prob > 0.45 else 0.12

    def _build_risk_context_text(self, macro_risk: dict[str, risk_model.RiskReport]) -> str:
        if not macro_risk:
            return "Macro risk unavailable."
        parts = [
            f"{symbol}: VaR {report.var_95:.2%}, loss prob {report.probability_of_loss:.2%}"
            for symbol, report in macro_risk.items()
        ]
        return " | ".join(parts)


class Product2Validator:
    """Validation and monitoring checks for Product 2 outputs."""

    def run_data_quality_checks(self, dataset: MarketDataset) -> dict[str, float]:
        return {
            "symbols": float(len(dataset.prices.columns)),
            "rows": float(len(dataset.prices)),
            "missing_price_ratio": float(dataset.prices.isna().mean().mean()),
            "missing_fundamentals_ratio": float(dataset.fundamentals.isna().mean().mean()),
            "recent_nonzero_volume_ratio": float((dataset.volumes.tail(20) > 0).mean().mean()),
        }

    def run_signal_sanity_checks(self, signals: list[SignalCandidate]) -> dict[str, float]:
        valid_sides = sum(1 for signal in signals if signal.side in {"LONG", "SHORT"})
        consistent_targets = sum(
            1
            for signal in signals
            if (signal.side == "LONG" and signal.target_price > signal.entry_price > signal.stop_price)
            or (signal.side == "SHORT" and signal.target_price < signal.entry_price < signal.stop_price)
        )
        return {
            "signals_total": float(len(signals)),
            "valid_sides": float(valid_sides),
            "consistent_targets": float(consistent_targets),
        }

    def run_forward_check(self, dataset: MarketDataset, horizon_days: int = 5) -> dict[str, float]:
        returns = dataset.prices.pct_change()
        momentum = dataset.prices.pct_change(21).shift(1)
        future = dataset.prices.pct_change(horizon_days).shift(-horizon_days)
        stack = pd.DataFrame(
            {
                "momentum": momentum.stack(dropna=False),
                "future": future.stack(dropna=False),
            }
        ).dropna()
        long_bucket = stack.nlargest(max(1, len(stack) // 10), "momentum")
        short_bucket = stack.nsmallest(max(1, len(stack) // 10), "momentum")
        return {
            "horizon_days": float(horizon_days),
            "avg_future_return_long_bucket": float(long_bucket["future"].mean()),
            "avg_future_return_short_bucket": float(short_bucket["future"].mean()),
            "coverage_rows": float(len(stack)),
            "daily_return_std": float(returns.stack(dropna=False).dropna().std(ddof=1)),
        }


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(numeric):
        return float("nan")
    return numeric


def _zscore(values: pd.Series) -> pd.Series:
    centered = values - values.mean()
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return centered / std
