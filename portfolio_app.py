"""Simple desktop dashboard for the classical Monte Carlo risk model."""

from __future__ import annotations

import io
import threading
import tkinter as tk
import webbrowser
from contextlib import redirect_stdout
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt

import monteCarloRisk as model


class RiskDashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Classical Monte Carlo Risk Lab")
        self.root.geometry("1400x1000")
        self.root.minsize(1050, 780)
        self.root.configure(bg="#f7f4ee")
        self.fields: list[tuple[tk.StringVar, tk.StringVar]] = []
        self.chart_canvases: list[FigureCanvasTkAgg] = []
        self.chart_paths: dict[str, str] = {}
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
        self.confidence_var = tk.StringVar(value="95")
        for row, (label, variable, suffix) in enumerate(
            [
                ("Starting portfolio value", self.value_var, "$"),
                ("Forecast horizon", self.horizon_var, "trading days"),
                ("Simulation paths", self.simulations_var, "paths"),
                ("Historical lookback", self.lookback_var, "years"),
                ("Risk confidence", self.confidence_var, "%"),
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

        self.results_tabs = ttk.Notebook(shell)
        self.results_tabs.pack(fill="both", expand=True)

        report_tab = ttk.Frame(self.results_tabs, padding=10)
        self.results_tabs.add(report_tab, text="Risk report")
        output_frame = ttk.LabelFrame(report_tab, text="Risk report", padding=10)
        output_frame.pack(fill="both", expand=True)
        report_scrollbar = ttk.Scrollbar(output_frame, orient="vertical")
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
            yscrollcommand=report_scrollbar.set,
        )
        report_scrollbar.configure(command=self.output.yview)
        report_scrollbar.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)
        self.output.insert("1.0", "Your portfolio report will appear here.\n")
        self.output.configure(state="disabled")

        chart_tab = ttk.Frame(self.results_tabs, padding=10)
        self.results_tabs.add(chart_tab, text="Charts")
        charts_frame = ttk.LabelFrame(chart_tab, text="Simulation charts", padding=10)
        charts_frame.pack(fill="both", expand=True)
        chart_controls = ttk.Frame(charts_frame)
        chart_controls.pack(fill="x", pady=(0, 8))
        self.portfolio_chart_button = ttk.Button(
            chart_controls,
            text="Open portfolio chart",
            command=lambda: self._open_chart("portfolio"),
            state="disabled",
        )
        self.portfolio_chart_button.pack(side="left")
        self.asset_chart_button = ttk.Button(
            chart_controls,
            text="Open asset charts",
            command=lambda: self._open_chart("assets"),
            state="disabled",
        )
        self.asset_chart_button.pack(side="left", padx=(8, 0))
        self.chart_canvas = tk.Canvas(charts_frame, bg="#f7f4ee", highlightthickness=0)
        chart_vertical_scrollbar = ttk.Scrollbar(charts_frame, orient="vertical", command=self.chart_canvas.yview)
        chart_horizontal_scrollbar = ttk.Scrollbar(charts_frame, orient="horizontal", command=self.chart_canvas.xview)
        self.chart_canvas.configure(
            yscrollcommand=chart_vertical_scrollbar.set,
            xscrollcommand=chart_horizontal_scrollbar.set,
        )
        self.chart_canvas.pack(side="left", fill="both", expand=True)
        chart_vertical_scrollbar.pack(side="right", fill="y")
        chart_horizontal_scrollbar.pack(side="bottom", fill="x")
        self.chart_content = ttk.Frame(self.chart_canvas)
        self.chart_window = self.chart_canvas.create_window((0, 0), window=self.chart_content, anchor="nw")
        self.chart_content.bind("<Configure>", self._update_chart_scrollregion)
        self.chart_canvas.bind("<Configure>", self._resize_chart_content)
        self._show_chart_placeholder()

        backtest_tab = ttk.Frame(self.results_tabs, padding=18)
        self.results_tabs.add(backtest_tab, text="Backtesting")
        backtest_tab.columnconfigure(1, weight=1)
        ttk.Label(
            backtest_tab,
            text="Rolling historical VaR check",
            style="Section.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(
            backtest_tab,
            text="Uses prior portfolio returns to test how often the next observed return breached the forecast threshold.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 16))
        self.backtest_window_var = tk.StringVar(value="252")
        self.backtest_confidence_var = tk.StringVar(value="95")
        ttk.Label(backtest_tab, text="Rolling window").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(backtest_tab, textvariable=self.backtest_window_var, width=18).grid(row=2, column=1, sticky="w", padx=18, pady=5)
        ttk.Label(backtest_tab, text="trading days", style="Subtitle.TLabel").grid(row=2, column=2, sticky="w")
        ttk.Label(backtest_tab, text="Confidence level").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(backtest_tab, textvariable=self.backtest_confidence_var, width=18).grid(row=3, column=1, sticky="w", padx=18, pady=5)
        ttk.Label(backtest_tab, text="%", style="Subtitle.TLabel").grid(row=3, column=2, sticky="w")
        self.backtest_button = ttk.Button(
            backtest_tab,
            text="Run backtest",
            style="Accent.TButton",
            command=self.run_backtest,
        )
        self.backtest_button.grid(row=4, column=0, columnspan=3, sticky="w", pady=(14, 12))
        self.backtest_output = tk.Text(
            backtest_tab,
            height=12,
            wrap="word",
            bg="#263238",
            fg="#f7f4ee",
            relief="flat",
            padx=14,
            pady=12,
            font=("Menlo", 10),
        )
        backtest_scrollbar = ttk.Scrollbar(backtest_tab, orient="vertical", command=self.backtest_output.yview)
        self.backtest_output.configure(yscrollcommand=backtest_scrollbar.set)
        self.backtest_output.grid(row=5, column=0, columnspan=2, sticky="nsew")
        backtest_scrollbar.grid(row=5, column=2, sticky="ns")
        backtest_tab.rowconfigure(5, weight=1)
        self.backtest_output.insert("1.0", "Backtest results will appear here.\n")
        self.backtest_output.configure(state="disabled")

    def _update_chart_scrollregion(self, _event: tk.Event) -> None:
        self.chart_canvas.configure(scrollregion=self.chart_canvas.bbox("all"))

    def _resize_chart_content(self, event: tk.Event) -> None:
        content_width = max(event.width, self.chart_content.winfo_reqwidth())
        self.chart_canvas.itemconfigure(self.chart_window, width=content_width)

    def _clear_chart_widgets(self) -> None:
        for canvas in self.chart_canvases:
            canvas.get_tk_widget().destroy()
            plt.close(canvas.figure)
        self.chart_canvases = []
        for child in self.chart_content.winfo_children():
            child.destroy()

    def _show_chart_placeholder(self) -> None:
        self._clear_chart_widgets()
        ttk.Label(self.chart_content, text="Charts will appear here after a simulation.", style="Subtitle.TLabel").pack(pady=45)

    def _open_chart(self, chart_name: str) -> None:
        chart_path = self.chart_paths.get(chart_name)
        if chart_path:
            webbrowser.open(Path(chart_path).resolve().as_uri())

    def _show_charts(self, figures: tuple[object, object], chart_paths: tuple[str, str]) -> None:
        self._clear_chart_widgets()
        self.chart_paths = {"portfolio": chart_paths[0], "assets": chart_paths[1]}
        self.portfolio_chart_button.configure(state="normal")
        self.asset_chart_button.configure(state="normal")
        for figure in figures:
            canvas = FigureCanvasTkAgg(figure, master=self.chart_content)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="x", pady=(0, 12))
            self.chart_canvases.append(canvas)
        self.chart_canvas.configure(scrollregion=self.chart_canvas.bbox("all"))
        self.results_tabs.select(1)

    def portfolio_string(self) -> str:
        entries: list[str] = []
        for row, (ticker_var, weight_var) in enumerate(self.fields, start=1):
            ticker = ticker_var.get().strip().upper()
            weight = weight_var.get().strip()
            if not ticker and not weight:
                continue
            if not ticker or not weight:
                raise ValueError(f"Holding row {row} needs both a ticker and a weight.")
            try:
                numeric_weight = float(weight)
            except ValueError as error:
                raise ValueError(f"Weight for holding row {row} must be a number; received '{weight}'.") from error
            entries.append(f"{ticker}:{numeric_weight / 100}")
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
            confidence = float(self.confidence_var.get()) / 100
            if portfolio_value <= 0 or horizon < 1 or simulations < 1 or lookback < 1:
                raise ValueError("Simulation settings must be positive.")
            if not 0.5 < confidence < 1:
                raise ValueError("Risk confidence must be between 50 and 100 percent.")
        except (ValueError, TypeError) as error:
            messagebox.showerror("Check your inputs", str(error))
            return

        self.run_button.configure(state="disabled")
        self.status.set("Downloading history and running simulation...")
        threading.Thread(
            target=self._run_worker,
            args=(portfolio, portfolio_value, horizon, simulations, lookback, confidence),
            daemon=True,
        ).start()

    def run_backtest(self) -> None:
        try:
            portfolio = model.parse_portfolio(self.portfolio_string())
            window = int(self.backtest_window_var.get())
            confidence = float(self.backtest_confidence_var.get()) / 100
            lookback = int(self.lookback_var.get())
            if window < 20 or lookback < 1:
                raise ValueError("Backtest window and historical lookback must be positive.")
            if not 0.5 < confidence < 1:
                raise ValueError("Confidence level must be between 50 and 100 percent.")
        except (ValueError, TypeError) as error:
            messagebox.showerror("Check your backtest inputs", str(error))
            return

        self.backtest_button.configure(state="disabled")
        self.status.set("Downloading history and running backtest...")
        threading.Thread(
            target=self._backtest_worker,
            args=(portfolio, window, confidence, lookback),
            daemon=True,
        ).start()

    def _backtest_worker(
        self,
        portfolio: dict[str, float],
        window: int,
        confidence: float,
        lookback: int,
    ) -> None:
        try:
            prices = model.download_portfolio_prices(list(portfolio), lookback)
            weights = np.array([portfolio[ticker] for ticker in prices.columns])
            result = model.backtest_portfolio(prices, weights, window, confidence)
            self.root.after(0, self._show_backtest_success, result, window, confidence)
        except Exception as error:  # Keep network/data errors visible in the dashboard.
            self.root.after(0, self._show_backtest_error, str(error))

    def _show_backtest_success(
        self,
        result: model.BacktestReport,
        window: int,
        confidence: float,
    ) -> None:
        output = (
            "ROLLING HISTORICAL BACKTEST\n"
            "===========================\n"
            f"Forecast window       {window} trading days\n"
            f"Confidence level      {confidence:.0%}\n"
            f"Test observations     {result.observations:,}\n"
            f"VaR breaches          {result.breaches:,}\n"
            f"Observed breach rate  {result.breach_rate:.2%}\n"
            f"Expected breach rate  {result.expected_breach_rate:.2%}\n"
            f"Average breach return {result.average_breach_return:+.2%}\n"
            f"Worst observed return {result.worst_return:+.2%}\n"
        )
        self.backtest_output.configure(state="normal")
        self.backtest_output.delete("1.0", "end")
        self.backtest_output.insert("1.0", output)
        self.backtest_output.configure(state="disabled")
        self.backtest_button.configure(state="normal")
        self.status.set("Backtest complete.")

    def _show_backtest_error(self, error: str) -> None:
        self.backtest_button.configure(state="normal")
        self.status.set("Backtest failed.")
        messagebox.showerror("Backtest error", error)

    def _run_worker(
        self,
        portfolio: dict[str, float],
        portfolio_value: float,
        horizon: int,
        simulations: int,
        lookback: int,
        confidence: float,
    ) -> None:
        try:
            prices = model.download_portfolio_prices(list(portfolio), lookback)
            weights = np.array([portfolio[ticker] for ticker in prices.columns])
            paths, asset_paths, log_returns = model.simulate_portfolio_paths(
                prices, weights, portfolio_value, horizon, simulations, seed=42
            )
            label = "Portfolio (" + ", ".join(portfolio) + ")"
            report = model.build_report(label, paths, log_returns, confidence)
            self.root.after(
                0,
                self._show_success,
                report,
                paths,
                asset_paths,
                prices,
                label,
                portfolio,
                horizon,
                simulations,
                confidence,
            )
        except Exception as error:  # Keep network/data errors visible in the dashboard.
            self.root.after(0, self._show_error, str(error))

    def _show_success(
        self,
        report: model.RiskReport,
        paths: np.ndarray,
        asset_paths: np.ndarray,
        prices: pd.DataFrame,
        label: str,
        portfolio: dict[str, float],
        horizon: int,
        simulations: int,
        confidence: float,
    ) -> None:
        output = str(Path(model.__file__).with_name("monte_carlo_risk.png"))
        asset_output = str(Path(model.__file__).with_name("portfolio_assets_risk.png"))
        portfolio_figure = model.plot_results(label, paths, report, output, confidence)
        asset_figure = model.plot_asset_results(list(prices.columns), prices, asset_paths, asset_output)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            model.print_report(
                report,
                horizon,
                simulations,
                confidence,
                output,
                portfolio,
                asset_output,
                include_output_paths=False,
            )
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", buffer.getvalue())
        self.output.configure(state="disabled")
        self._show_charts((portfolio_figure, asset_figure), (output, asset_output))
        self.status.set("Simulation complete. Live charts are available in the Charts tab.")
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
        self._show_chart_placeholder()
        self.chart_paths = {}
        self.portfolio_chart_button.configure(state="disabled")
        self.asset_chart_button.configure(state="disabled")
        self.backtest_output.configure(state="normal")
        self.backtest_output.delete("1.0", "end")
        self.backtest_output.insert("1.0", "Backtest results will appear here.\n")
        self.backtest_output.configure(state="disabled")
        self.status.set("Ready. Configure your portfolio and run a simulation.")


if __name__ == "__main__":
    app_root = tk.Tk()
    RiskDashboard(app_root)
    app_root.mainloop()
