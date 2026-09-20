"""Simple desktop dashboard for the classical Monte Carlo risk model."""

from __future__ import annotations

import io
import threading
import tkinter as tk
from contextlib import redirect_stdout
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

import matplotlib

matplotlib.use("Agg")

import monteCarloRisk as model


class RiskDashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Classical Monte Carlo Risk Lab")
        self.root.geometry("760x700")
        self.root.minsize(700, 620)
        self.root.configure(bg="#f7f4ee")
        self.fields: list[tuple[tk.StringVar, tk.StringVar]] = []
        self.status = tk.StringVar(value="Ready. Configure your portfolio and run a simulation.")
        self.build_interface()

    def build_interface(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f7f4ee")
        style.configure("TLabel", background="#f7f4ee", foreground="#263238", font=("Avenir Next", 11))
        style.configure("Title.TLabel", font=("Avenir Next", 24, "bold"), foreground="#263238")
        style.configure("Subtitle.TLabel", font=("Avenir Next", 11), foreground="#52666a")
        style.configure("Section.TLabel", font=("Avenir Next", 13, "bold"), foreground="#2f6f73")
        style.configure("Accent.TButton", font=("Avenir Next", 12, "bold"), foreground="white", background="#c85c3d", padding=10)
        style.map("Accent.TButton", background=[("active", "#a94d34")])

        shell = ttk.Frame(self.root, padding=28)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="Classical Monte Carlo Risk Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            shell,
            text="Edit the portfolio here. No command-line formatting required.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 22))

        portfolio_frame = ttk.LabelFrame(shell, text="Portfolio holdings", padding=16)
        portfolio_frame.pack(fill="x")
        ttk.Label(portfolio_frame, text="Ticker").grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Label(portfolio_frame, text="Weight").grid(row=0, column=1, sticky="w")

        defaults = [("SPY", "30"), ("AAPL", "20"), ("MSFT", "20"), ("GOOGL", "15"), ("AMZN", "15")]
        for row, (ticker, weight) in enumerate(defaults, start=1):
            ticker_var = tk.StringVar(value=ticker)
            weight_var = tk.StringVar(value=weight)
            ttk.Entry(portfolio_frame, textvariable=ticker_var, width=16).grid(row=row, column=0, padx=(0, 12), pady=4, sticky="w")
            ttk.Entry(portfolio_frame, textvariable=weight_var, width=16).grid(row=row, column=1, pady=4, sticky="w")
            ttk.Label(portfolio_frame, text="%").grid(row=row, column=2, padx=(6, 0), sticky="w")
            self.fields.append((ticker_var, weight_var))

        ttk.Label(
            portfolio_frame,
            text="Weights are percentages and should total 100%. Leave a row blank to use fewer holdings.",
            style="Subtitle.TLabel",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 0))

        settings = ttk.LabelFrame(shell, text="Simulation settings", padding=16)
        settings.pack(fill="x", pady=18)
        settings.columnconfigure(1, weight=1)
        self.value_var = tk.StringVar(value="100000")
        self.horizon_var = tk.StringVar(value="252")
        self.simulations_var = tk.StringVar(value="20000")
        self.lookback_var = tk.StringVar(value="5")
        for row, (label, variable, suffix) in enumerate(
            [
                ("Starting portfolio value", self.value_var, "$"),
                ("Forecast horizon", self.horizon_var, "trading days"),
                ("Simulation paths", self.simulations_var, "paths"),
                ("Historical lookback", self.lookback_var, "years"),
            ]
        ):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(settings, textvariable=variable, width=18).grid(row=row, column=1, sticky="w", padx=18, pady=5)
            ttk.Label(settings, text=suffix, style="Subtitle.TLabel").grid(row=row, column=2, sticky="w")

        controls = ttk.Frame(shell)
        controls.pack(fill="x", pady=(0, 12))
        self.run_button = ttk.Button(controls, text="Run classical Monte Carlo", style="Accent.TButton", command=self.run_simulation)
        self.run_button.pack(side="left")
        ttk.Button(controls, text="Clear report", command=self.clear_report).pack(side="left", padx=10)
        ttk.Label(controls, textvariable=self.status, style="Subtitle.TLabel").pack(side="left", padx=10)

        output_frame = ttk.LabelFrame(shell, text="Risk report", padding=10)
        output_frame.pack(fill="both", expand=True)
        self.output = tk.Text(
            output_frame,
            height=12,
            wrap="none",
            bg="#263238",
            fg="#f7f4ee",
            insertbackground="#f7f4ee",
            relief="flat",
            padx=14,
            pady=12,
            font=("Menlo", 10),
        )
        self.output.pack(fill="both", expand=True)
        self.output.insert("1.0", "Your portfolio report will appear here.\n")
        self.output.configure(state="disabled")

    def portfolio_string(self) -> str:
        entries: list[str] = []
        for ticker_var, weight_var in self.fields:
            ticker = ticker_var.get().strip().upper()
            weight = weight_var.get().strip()
            if not ticker and not weight:
                continue
            if not ticker or not weight:
                raise ValueError("Each active row needs both a ticker and a weight.")
            entries.append(f"{ticker}:{float(weight) / 100}")
        if not entries:
            raise ValueError("Add at least one holding.")
        return ",".join(entries)

    def run_simulation(self) -> None:
        try:
            portfolio = model.parse_portfolio(self.portfolio_string())
            portfolio_value = float(self.value_var.get())
            horizon = int(self.horizon_var.get())
            simulations = int(self.simulations_var.get())
            lookback = int(self.lookback_var.get())
            if portfolio_value <= 0 or horizon < 1 or simulations < 1 or lookback < 1:
                raise ValueError("Simulation settings must be positive.")
        except (ValueError, TypeError) as error:
            messagebox.showerror("Check your inputs", str(error))
            return

        self.run_button.configure(state="disabled")
        self.status.set("Downloading history and running simulation...")
        threading.Thread(
            target=self._run_worker,
            args=(portfolio, portfolio_value, horizon, simulations, lookback),
            daemon=True,
        ).start()

    def _run_worker(self, portfolio: dict[str, float], portfolio_value: float, horizon: int, simulations: int, lookback: int) -> None:
        try:
            prices = model.download_portfolio_prices(list(portfolio), lookback)
            weights = np.array([portfolio[ticker] for ticker in prices.columns])
            paths, asset_paths, log_returns = model.simulate_portfolio_paths(
                prices, weights, portfolio_value, horizon, simulations, seed=42
            )
            asset_output = "portfolio_assets_risk.png"
            model.plot_asset_results(list(prices.columns), prices, asset_paths, asset_output)
            output = "monte_carlo_risk.png"
            label = "Portfolio (" + ", ".join(portfolio) + ")"
            report = model.build_report(label, paths, log_returns, 0.95)
            model.plot_results(label, paths, report, output)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                model.print_report(report, horizon, simulations, 0.95, output, portfolio, asset_output)
            self.root.after(0, self._show_success, buffer.getvalue())
        except Exception as error:  # Keep network/data errors visible in the dashboard.
            self.root.after(0, self._show_error, str(error))

    def _show_success(self, report: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", report)
        self.output.configure(state="disabled")
        self.status.set("Simulation complete. Charts saved in the project folder.")
        self.run_button.configure(state="normal")

    def _show_error(self, error: str) -> None:
        self.status.set("Simulation failed.")
        self.run_button.configure(state="normal")
        messagebox.showerror("Simulation error", error)

    def clear_report(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", "Your portfolio report will appear here.\n")
        self.output.configure(state="disabled")
        self.status.set("Ready. Configure your portfolio and run a simulation.")


if __name__ == "__main__":
    app_root = tk.Tk()
    RiskDashboard(app_root)
    app_root.mainloop()
