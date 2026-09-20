# Steelhacks 2026

Hi, I'm Caden.

I am a junior at the University of Pittsburgh studying **Finance** and **Computer Science**. I am interested in the intersection of markets, technology, and analytical problem solving, especially when a project turns financial ideas into something useful and measurable.

## What I am working on

- Quantitative finance and market-risk analysis
- Monte Carlo simulation and financial modeling
- Data-driven applications with Python
- Building practical projects through hackathons and independent experimentation

## Featured project

### Classical Monte Carlo Risk Lab

A Python-based market-risk simulator that uses historical Yahoo Finance data and geometric Brownian motion to model potential price paths. The analysis includes:

- Simulated price distributions
- Annualized volatility
- Probability of loss
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Confidence-band visualizations

## Run the project

For the streamlined desktop interface, run:

```bash
./.venv/bin/python portfolio_app.py
```

The dashboard lets you edit up to five tickers and their percentage weights, set the starting account value, choose the historical lookback, forecast horizon, simulation paths, and VaR/CVaR confidence level. Click **Run classical Monte Carlo** to download market history, run the same correlated simulation used by the CLI, and display the report and both live Matplotlib charts directly in the app's **Charts** tab. The chart files are also exported to the project folder and can be opened with the chart buttons.

The **Backtesting** tab has its own confidence setting and runs a rolling historical VaR check. It uses the preceding rolling window of weighted portfolio returns to set a historical loss threshold, then counts how often the next observed return breaches that threshold. The output shows the observed breach rate, expected breach rate, number of test observations, average breach return, and worst observed return.

The original command-line interface remains available for reproducible runs and automation:

```bash
./.venv/bin/python monteCarloRisk.py
```

By default, the script runs the five-stock example portfolio: SPY, AAPL, MSFT, GOOGL, and AMZN. Run it with:

```bash
python monteCarloRisk.py
```

For a one-year, 20,000-path single-asset simulation, pass `--ticker`:

```bash
python monteCarloRisk.py --ticker SPY --horizon 252 --simulations 20000
```

Do not combine `--ticker` and `--portfolio`; `--ticker` is the single-stock mode and `--portfolio` is the multi-stock mode.

To model a portfolio, provide decimal weights that add up to `1.0`:

```bash
python monteCarloRisk.py \
	--portfolio SPY:0.60,AAPL:0.25,MSFT:0.15 \
	--horizon 252 \
	--simulations 20000
```

Portfolio mode supports up to five stocks, estimates historical correlations between holdings, and applies correlated Monte Carlo shocks. It saves both the combined portfolio chart and a five-panel `portfolio_assets_risk.png` chart for the individual holdings:

```bash
python monteCarloRisk.py \
	--portfolio SPY:0.30,AAPL:0.20,MSFT:0.20,GOOGL:0.15,AMZN:0.15 \
	--horizon 252 \
	--simulations 20000
```

The portfolio starts at `$100,000` by default. Weights are converted into share quantities using each stock's latest price, so the simulated portfolio value and risk metrics are in dollars. Change the notional with `--portfolio-value 250000`.

The terminal report lists every ticker and its allocation, followed by portfolio-level volatility, probability of loss, VaR, and CVaR. The two chart files are:

- `monte_carlo_risk.png`: combined portfolio paths and terminal-value distribution
- `portfolio_assets_risk.png`: simulated paths for each individual holding

Weights are decimal portfolio allocations and must add up to `1.0`. For example, `SPY:0.30` means 30% of the modeled portfolio. The model does not connect to brokerage credentials or read private account data.

The horizon is measured in trading days: `21` is about one month, `63` is about one quarter, and `252` is about one trading year.

## Backtesting methodology

To backtest the model, use a walk-forward process rather than fitting on the entire history:

1. Choose a historical training window, such as the prior 5 years.
2. Estimate each holding's drift, volatility, and covariance using only that window.
3. Simulate the next forecast horizon, such as 21 trading days.
4. Compare the simulated return distribution with the actual realized portfolio return.
5. Move the window forward and repeat across history.

The key diagnostics are VaR breach rate, expected shortfall accuracy, average forecast error, and whether the realized losses fall inside the simulated confidence bands. A 95% VaR model should be breached roughly 5% of the time over many independent test periods, acknowledging that market regimes change.

## Tools and interests

**Technical:** Python, NumPy, pandas, Matplotlib, yfinance, quantitative analysis

**Beyond code:** Personal investing, classic literature, movies, cars, and motorcycles

## A little more about me

I enjoy learning across disciplines. Finance gives me a way to think about uncertainty and incentives, while computer science gives me the tools to test ideas and build systems around them. Outside of school and projects, I am usually reading a classic novel, watching a great film, or learning something new about cars and motorcycles.

I am always interested in meeting people working at the intersection of finance, technology, and data.
