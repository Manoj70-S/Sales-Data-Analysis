"""Matplotlib chart generator for executive sales visualizations."""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for reliable headless saving
import matplotlib.pyplot as plt
import numpy as np

from .analytics import AnalysisResult


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#cccccc",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "font.size": 10,
        }
    )


def generate_charts(result: AnalysisResult, charts_dir: str | Path) -> list[Path]:
    """Generate and save all sales analytics charts, returning saved file paths."""
    _configure_style()
    out_dir = Path(charts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    saved.append(_plot_monthly_revenue_profit_forecast(result, out_dir / "monthly_revenue_profit.png"))
    saved.append(_plot_monthly_growth(result, out_dir / "monthly_growth.png"))
    saved.append(_plot_product_pareto(result, out_dir / "product_pareto.png"))
    saved.append(_plot_regional_heatmap(result, out_dir / "regional_profit_heatmap.png"))
    return saved


def _plot_monthly_revenue_profit_forecast(result: AnalysisResult, path: Path) -> Path:
    m = result.monthly
    f = result.forecast
    all_months = m.months + f.future_months
    x_hist = np.arange(len(m.months))
    x_fut = np.arange(len(m.months), len(all_months))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(x_hist, m.net_revenue / 1000.0, marker="o", linewidth=2.2, color="#1f77b4", label="Net Revenue ($K)")
    ax.plot(x_hist, m.net_profit / 1000.0, marker="s", linewidth=2.2, color="#2ca02c", label="Net Profit ($K)")

    valid_ma = ~np.isnan(m.revenue_ma3)
    if np.any(valid_ma):
        ax.plot(
            x_hist[valid_ma],
            m.revenue_ma3[valid_ma] / 1000.0,
            linestyle="--",
            linewidth=1.8,
            color="#ff7f0e",
            label="3-Month Moving Avg ($K)",
        )

    if len(f.future_months) > 0:
        # Connect last historical point to first forecast point
        x_conn = np.concatenate([[x_hist[-1]], x_fut])
        y_rev_conn = np.concatenate([[m.net_revenue[-1]], f.predicted_revenue]) / 1000.0
        y_prof_conn = np.concatenate([[m.net_profit[-1]], f.predicted_profit]) / 1000.0

        ax.plot(x_conn, y_rev_conn, linestyle=":", marker="^", linewidth=2.0, color="#1f77b4", alpha=0.75, label="Forecast Revenue ($K)")
        ax.plot(x_conn, y_prof_conn, linestyle=":", marker="^", linewidth=2.0, color="#2ca02c", alpha=0.75, label="Forecast Profit ($K)")
        ax.axvline(x_hist[-1] + 0.5, color="#888888", linestyle="-.", alpha=0.7)

    ax.set_title("Monthly Net Revenue, Net Profit & Forecast", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("USD (Thousands)")
    ax.set_xticks(np.arange(len(all_months)))
    ax.set_xticklabels(all_months, rotation=45, ha="right")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_monthly_growth(result: AnalysisResult, path: Path) -> Path:
    m = result.monthly
    fig, ax = plt.subplots(figsize=(10, 4.8))

    growth = np.nan_to_num(m.mom_revenue_growth_pct, nan=0.0)
    colors = ["#2ca02c" if val >= 0 else "#d62728" for val in growth]
    x = np.arange(len(m.months))

    bars = ax.bar(x, growth, color=colors, alpha=0.85, edgecolor="#333333", linewidth=0.5)
    ax.axhline(0, color="#333333", linewidth=0.9)

    for idx, (bar, raw_val) in enumerate(zip(bars, m.mom_revenue_growth_pct)):
        if np.isnan(raw_val):
            continue
        y_pos = raw_val + (1.2 if raw_val >= 0 else -2.2)
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{raw_val:+.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_title("Month-over-Month (MoM) Net Revenue Growth %", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Growth (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(m.months, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_product_pareto(result: AnalysisResult, path: Path) -> Path:
    p = result.products
    order = p.rank_by_revenue

    names = [p.products[i] for i in order]
    revs = p.net_revenue[order] / 1000.0
    profs = p.net_profit[order] / 1000.0
    cum_pct = p.cumulative_revenue_share_pct[order]

    x = np.arange(len(names))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(x - width / 2, revs, width=width, color="#1f77b4", label="Net Revenue ($K)")
    ax1.bar(x + width / 2, profs, width=width, color="#2ca02c", label="Net Profit ($K)")
    ax1.set_ylabel("USD (Thousands)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, cum_pct, color="#d62728", marker="o", linewidth=2.0, label="Cumulative Revenue %")
    ax2.axhline(80.0, color="#d62728", linestyle="--", alpha=0.6, label="80% Pareto Threshold")
    ax2.set_ylabel("Cumulative Revenue Share (%)")
    ax2.set_ylim(0, 108)
    ax2.grid(False)

    ax1.set_title("Product Revenue vs. Profit & ABC Pareto Curve", fontsize=13, fontweight="bold")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_regional_heatmap(result: AnalysisResult, path: Path) -> Path:
    p = result.products
    r = result.regional
    order = p.rank_by_profit
    matrix = r.product_region_profit_matrix[order, :] / 1000.0
    prod_labels = [p.products[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.45 * len(prod_labels) + 1.5)))
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto")

    ax.set_xticks(np.arange(len(r.regions)))
    ax.set_xticklabels(r.regions)
    ax.set_yticks(np.arange(len(prod_labels)))
    ax.set_yticklabels(prod_labels)

    max_val = float(np.max(matrix)) if matrix.size > 0 else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt_color = "white" if val > max_val * 0.6 else "#111111"
            ax.text(j, i, f"${val:,.1f}K", ha="center", va="center", color=txt_color, fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Net Profit ($K)")
    ax.set_title("Product x Region Net Profit Heatmap ($K)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
