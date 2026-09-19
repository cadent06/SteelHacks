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

Run a one-year, 20,000-path single-asset simulation with:

```bash
python monteCarloRisk.py --ticker SPY --horizon 252 --simulations 20000
```

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

The terminal report lists every ticker and its allocation, followed by portfolio-level volatility, probability of loss, VaR, and CVaR. The two chart files are:

- `monte_carlo_risk.png`: combined portfolio paths and terminal-value distribution
- `portfolio_assets_risk.png`: simulated paths for each individual holding

Weights are decimal portfolio allocations and must add up to `1.0`. For example, `SPY:0.30` means 30% of the modeled portfolio. The current model normalizes the portfolio value from these weights; it does not connect to brokerage credentials or read private account data.

The horizon is measured in trading days: `21` is about one month, `63` is about one quarter, and `252` is about one trading year.

## Tools and interests

**Technical:** Python, NumPy, pandas, Matplotlib, yfinance, quantitative analysis

**Beyond code:** Personal investing, classic literature, movies, cars, and motorcycles

## A little more about me

I enjoy learning across disciplines. Finance gives me a way to think about uncertainty and incentives, while computer science gives me the tools to test ideas and build systems around them. Outside of school and projects, I am usually reading a classic novel, watching a great film, or learning something new about cars and motorcycles.

I am always interested in meeting people working at the intersection of finance, technology, and data.
