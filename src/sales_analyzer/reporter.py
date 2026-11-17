"""Console summary, CSV exports, and executive Markdown report generator."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .analytics import AnalysisResult
from .loader import SalesDataset


def _fmt_money(val: float) -> str:
    return f"${val:,.2f}"


def _fmt_pct(val: float, signed: bool = False) -> str:
    if np.isnan(val):
        return "N/A"
    return f"{val:+.2f}%" if signed else f"{val:.2f}%"


def format_console_summary(dataset: SalesDataset, result: AnalysisResult) -> str:
    """Create a concise, executive-ready terminal summary."""
    o = result.overall
    m = result.monthly
    p = result.products
    r = result.regional
    c = dataset.cleaning_report

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("SALES DATA ANALYZER - EXECUTIVE SUMMARY")
    lines.append("=" * 72)
    lines.append(
        f"Dataset Shape  : {len(dataset.months)} months x {len(dataset.products)} products x {len(dataset.regions)} regions"
    )
    lines.append(
        f"Data Cleaning  : {c.total_rows_read} rows read | {c.duplicate_rows_removed} duplicates removed | "
        f"{sum(c.missing_values_imputed.values())} missing imputed | {sum(c.invalid_values_fixed.values())} invalid fixed"
    )
    lines.append("-" * 72)
    lines.append("OVERALL KPIs")
    lines.append(f"  Gross Revenue     : {_fmt_money(o.gross_revenue)}")
    lines.append(f"  Discounts Given   : -{_fmt_money(o.total_discounts)}")
    lines.append(f"  Refunds (Returns) : -{_fmt_money(o.refund_value)}  (Return Rate: {_fmt_pct(o.return_rate_pct)})")
    lines.append(f"  Net Revenue       : {_fmt_money(o.net_revenue)}")
    lines.append(f"  COGS + Shipping   : -{_fmt_money(o.total_cogs + o.total_shipping)}")
    lines.append(f"  Net Profit        : {_fmt_money(o.net_profit)}  (Margin: {_fmt_pct(o.profit_margin_pct)})")
    lines.append(f"  Net Units Sold    : {int(round(o.net_units_sold)):,}  (Avg Selling Price: {_fmt_money(o.avg_selling_price)})")
    lines.append(f"  Compound Monthly Growth (CMGR): {_fmt_pct(o.cagr_monthly_pct, signed=True)}")
    lines.append("-" * 72)
    lines.append("MONTHLY HIGHLIGHTS")
    lines.append(f"  Best Revenue Month  : {m.best_month} ({_fmt_money(float(np.max(m.net_revenue)))})")
    lines.append(f"  Worst Revenue Month : {m.worst_month} ({_fmt_money(float(np.min(m.net_revenue)))})")
    if m.best_growth_month:
        lines.append(
            f"  Strongest MoM Surge : {m.best_growth_month} ({_fmt_pct(float(np.nanmax(m.mom_revenue_growth_pct)), signed=True)})"
        )
        lines.append(
            f"  Steepest MoM Drop   : {m.worst_growth_month} ({_fmt_pct(float(np.nanmin(m.mom_revenue_growth_pct)), signed=True)})"
        )
    lines.append("-" * 72)
    lines.append("TOP 5 PRODUCTS BY NET PROFIT (with ABC Pareto Tier)")
    for rank, idx in enumerate(p.rank_by_profit[:5], start=1):
        i = int(idx)
        lines.append(
            f"  {rank}. {p.products[i]:<24} | Tier {p.abc_tier[i]} | "
            f"Rev: {_fmt_money(float(p.net_revenue[i])):>12} | "
            f"Profit: {_fmt_money(float(p.net_profit[i])):>11} | "
            f"Margin: {_fmt_pct(float(p.profit_margin_pct[i])):>6}"
        )

    if p.at_risk_products:
        lines.append(f"  At-Risk / Watchlist : {', '.join(p.at_risk_products)}")

    lines.append("-" * 72)
    lines.append("REGIONAL PERFORMANCE")
    reg_order = np.argsort(r.net_profit)[::-1]
    for idx in reg_order:
        i = int(idx)
        lines.append(
            f"  {r.regions[i]:<10} | Rev: {_fmt_money(float(r.net_revenue[i])):>12} ({_fmt_pct(float(r.revenue_share_pct[i])):>6}) | "
            f"Profit: {_fmt_money(float(r.net_profit[i])):>11} | Margin: {_fmt_pct(float(r.profit_margin_pct[i])):>6}"
        )

    lines.append("-" * 72)
    lines.append(f"ANOMALIES DETECTED ({len(result.anomalies)})")
    if result.anomalies:
        for a in result.anomalies[:5]:
            lines.append(
                f"  [{a.direction.upper():<5}] {a.month} | {a.product} ({a.region}) | "
                f"Rev: {_fmt_money(a.net_revenue)} | Z={a.z_score:+.2f}"
            )
    else:
        lines.append("  None detected above threshold.")

    lines.append("-" * 72)
    lines.append(f"FORECAST (Next {len(result.forecast.future_months)} Months via Linear Trend)")
    for fm, fr, fp in zip(
        result.forecast.future_months,
        result.forecast.predicted_revenue,
        result.forecast.predicted_profit,
    ):
        lines.append(f"  {fm} | Projected Revenue: {_fmt_money(float(fr)):>12} | Projected Profit: {_fmt_money(float(fp)):>11}")

    lines.append("-" * 72)
    lines.append("WHAT-IF SCENARIO MODELING")
    for s in result.scenarios:
        lines.append(
            f"  {s.label:<45} | Profit: {_fmt_money(s.scenario_profit):>12} "
            f"({s.profit_delta:+,.2f} / {s.profit_delta_pct:+.1f}%)"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def export_csv_summaries(result: AnalysisResult, output_dir: str | Path) -> list[Path]:
    """Export monthly, product, and regional summaries to CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []

    # 1. Monthly summary CSV
    m = result.monthly
    monthly_path = out / "monthly_summary.csv"
    with monthly_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "month",
                "net_units",
                "gross_revenue",
                "net_revenue",
                "net_profit",
                "profit_margin_pct",
                "return_rate_pct",
                "mom_revenue_growth_pct",
                "mom_profit_growth_pct",
                "revenue_ma3",
            ]
        )
        for i, month in enumerate(m.months):
            writer.writerow(
                [
                    month,
                    round(float(m.net_units[i]), 2),
                    round(float(m.gross_revenue[i]), 2),
                    round(float(m.net_revenue[i]), 2),
                    round(float(m.net_profit[i]), 2),
                    round(float(m.profit_margin_pct[i]), 2),
                    round(float(m.return_rate_pct[i]), 2),
                    "" if np.isnan(m.mom_revenue_growth_pct[i]) else round(float(m.mom_revenue_growth_pct[i]), 2),
                    "" if np.isnan(m.mom_profit_growth_pct[i]) else round(float(m.mom_profit_growth_pct[i]), 2),
                    "" if np.isnan(m.revenue_ma3[i]) else round(float(m.revenue_ma3[i]), 2),
                ]
            )
    exported.append(monthly_path)

    # 2. Product summary CSV (sorted by net revenue desc)
    p = result.products
    product_path = out / "product_summary.csv"
    with product_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "product",
                "category",
                "abc_tier",
                "net_units",
                "net_revenue",
                "net_profit",
                "profit_margin_pct",
                "return_rate_pct",
                "revenue_share_pct",
                "cumulative_revenue_share_pct",
                "monthly_trend_slope_usd",
            ]
        )
        for idx in p.rank_by_revenue:
            i = int(idx)
            writer.writerow(
                [
                    p.products[i],
                    p.categories[i],
                    p.abc_tier[i],
                    round(float(p.net_units[i]), 2),
                    round(float(p.net_revenue[i]), 2),
                    round(float(p.net_profit[i]), 2),
                    round(float(p.profit_margin_pct[i]), 2),
                    round(float(p.return_rate_pct[i]), 2),
                    round(float(p.revenue_share_pct[i]), 2),
                    round(float(p.cumulative_revenue_share_pct[i]), 2),
                    round(float(p.revenue_trend_slope[i]), 2),
                ]
            )
    exported.append(product_path)

    # 3. Regional summary CSV
    r = result.regional
    regional_path = out / "regional_summary.csv"
    with regional_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["region", "net_units", "net_revenue", "net_profit", "profit_margin_pct", "revenue_share_pct"]
        )
        for i, region in enumerate(r.regions):
            writer.writerow(
                [
                    region,
                    round(float(r.net_units[i]), 2),
                    round(float(r.net_revenue[i]), 2),
                    round(float(r.net_profit[i]), 2),
                    round(float(r.profit_margin_pct[i]), 2),
                    round(float(r.revenue_share_pct[i]), 2),
                ]
            )
    exported.append(regional_path)

    return exported


def export_markdown_report(
    dataset: SalesDataset,
    result: AnalysisResult,
    report_path: str | Path,
    chart_paths: list[Path] | None = None,
) -> Path:
    """Write a complete Markdown executive report."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    o = result.overall
    m = result.monthly
    p = result.products
    r = result.regional
    c = dataset.cleaning_report

    md: list[str] = []
    md.append("# Sales Data Analyzer — Executive Report\n")
    md.append(
        f"**Scope:** {len(dataset.months)} months (`{dataset.months[0]}` to `{dataset.months[-1]}`), "
        f"{len(dataset.products)} products, {len(dataset.regions)} regions.\n"
    )

    md.append("## 1. Data Quality & Cleaning Audit\n")
    md.append(f"- **Rows Read:** {c.total_rows_read}")
    md.append(f"- **Duplicates Removed:** {c.duplicate_rows_removed}")
    md.append(f"- **Missing Values Imputed:** {c.missing_values_imputed or 'None'}")
    md.append(f"- **Invalid Values Corrected:** {c.invalid_values_fixed or 'None'}\n")

    md.append("## 2. Overall Financial KPIs\n")
    md.append("| Metric | Value |")
    md.append("| --- | ---: |")
    md.append(f"| Gross Revenue | {_fmt_money(o.gross_revenue)} |")
    md.append(f"| Discounts | -{_fmt_money(o.total_discounts)} |")
    md.append(f"| Refunds (Returns) | -{_fmt_money(o.refund_value)} |")
    md.append(f"| **Net Revenue** | **{_fmt_money(o.net_revenue)}** |")
    md.append(f"| COGS | -{_fmt_money(o.total_cogs)} |")
    md.append(f"| Shipping Costs | -{_fmt_money(o.total_shipping)} |")
    md.append(f"| **Net Profit** | **{_fmt_money(o.net_profit)}** |")
    md.append(f"| Profit Margin | {_fmt_pct(o.profit_margin_pct)} |")
    md.append(f"| Net Units Sold | {int(round(o.net_units_sold)):,} |")
    md.append(f"| Return Rate | {_fmt_pct(o.return_rate_pct)} |")
    md.append(f"| Average Selling Price (ASP) | {_fmt_money(o.avg_selling_price)} |")
    md.append(f"| Compound Monthly Growth (CMGR) | {_fmt_pct(o.cagr_monthly_pct, signed=True)} |\n")

    md.append("## 3. Monthly Performance & MoM Growth\n")
    md.append("| Month | Net Units | Net Revenue | Net Profit | Margin % | MoM Rev Growth | 3M Moving Avg |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i, month in enumerate(m.months):
        ma_str = "N/A" if np.isnan(m.revenue_ma3[i]) else _fmt_money(float(m.revenue_ma3[i]))
        md.append(
            f"| {month} | {int(round(float(m.net_units[i]))):,} | {_fmt_money(float(m.net_revenue[i]))} | "
            f"{_fmt_money(float(m.net_profit[i]))} | {_fmt_pct(float(m.profit_margin_pct[i]))} | "
            f"{_fmt_pct(float(m.mom_revenue_growth_pct[i]), signed=True)} | {ma_str} |"
        )
    md.append("")

    md.append("## 4. Product Performance & ABC (Pareto 80/20) Ranking\n")
    md.append("| Rank | Product | Category | ABC | Net Units | Net Revenue | Net Profit | Margin % | Cum Rev % | Monthly Trend |")
    md.append("| ---: | --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, idx in enumerate(p.rank_by_revenue, start=1):
        i = int(idx)
        md.append(
            f"| {rank} | {p.products[i]} | {p.categories[i]} | **{p.abc_tier[i]}** | "
            f"{int(round(float(p.net_units[i]))):,} | {_fmt_money(float(p.net_revenue[i]))} | "
            f"{_fmt_money(float(p.net_profit[i]))} | {_fmt_pct(float(p.profit_margin_pct[i]))} | "
            f"{_fmt_pct(float(p.cumulative_revenue_share_pct[i]))} | {p.revenue_trend_slope[i]:+,.1f}/mo |"
        )
    if p.at_risk_products:
        md.append(f"\n> **Watchlist (Declining trend, margin <20%, or return rate >6%):** {', '.join(p.at_risk_products)}\n")
    else:
        md.append("")

    md.append("## 5. Regional Breakdown\n")
    md.append("| Region | Net Units | Net Revenue | Share % | Net Profit | Margin % |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for idx in np.argsort(r.net_profit)[::-1]:
        i = int(idx)
        md.append(
            f"| {r.regions[i]} | {int(round(float(r.net_units[i]))):,} | {_fmt_money(float(r.net_revenue[i]))} | "
            f"{_fmt_pct(float(r.revenue_share_pct[i]))} | {_fmt_money(float(r.net_profit[i]))} | "
            f"{_fmt_pct(float(r.profit_margin_pct[i]))} |"
        )
    md.append("")

    md.append("## 6. Anomalies Detected\n")
    if result.anomalies:
        md.append("| Month | Product | Region | Type | Net Revenue | Z-Score |")
        md.append("| --- | --- | --- | :---: | ---: | ---: |")
        for a in result.anomalies:
            md.append(
                f"| {a.month} | {a.product} | {a.region} | {a.direction.upper()} | "
                f"{_fmt_money(a.net_revenue)} | {a.z_score:+.2f} |"
            )
    else:
        md.append("No anomalies exceeded the Z-score threshold.")
    md.append("")

    md.append("## 7. Linear Trend Forecast\n")
    md.append(
        f"Estimated linear trend: **{_fmt_money(result.forecast.revenue_slope_per_month)}/month** revenue, "
        f"**{_fmt_money(result.forecast.profit_slope_per_month)}/month** profit.\n"
    )
    md.append("| Forecast Month | Projected Net Revenue | Projected Net Profit |")
    md.append("| --- | ---: | ---: |")
    for fm, fr, fp in zip(
        result.forecast.future_months,
        result.forecast.predicted_revenue,
        result.forecast.predicted_profit,
    ):
        md.append(f"| {fm} | {_fmt_money(float(fr))} | {_fmt_money(float(fp))} |")
    md.append("")

    if result.top_correlations:
        md.append("## 8. Top Co-Moving Product Pairs (Monthly Revenue Correlation)\n")
        md.append("| Product A | Product B | Pearson r |")
        md.append("| --- | --- | ---: |")
        for pair in result.top_correlations:
            md.append(f"| {pair.product_a} | {pair.product_b} | {pair.correlation:+.3f} |")
        md.append("")

    md.append("## 9. What-If Scenario Simulations\n")
    md.append("| Scenario | Projected Revenue | Rev Delta % | Projected Profit | Profit Delta | Profit Delta % | Margin % |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in result.scenarios:
        md.append(
            f"| {s.label} | {_fmt_money(s.scenario_revenue)} | {_fmt_pct(s.revenue_delta_pct, signed=True)} | "
            f"{_fmt_money(s.scenario_profit)} | {s.profit_delta:+,.2f} | "
            f"{_fmt_pct(s.profit_delta_pct, signed=True)} | {_fmt_pct(s.scenario_margin_pct)} |"
        )
    md.append("")

    if chart_paths:
        md.append("## 10. Visual Charts\n")
        for cp in chart_paths:
            rel = cp.relative_to(path.parent).as_posix() if cp.is_relative_to(path.parent) else cp.name
            md.append(f"### {cp.stem.replace('_', ' ').title()}\n")
            md.append(f"![{cp.stem}]({rel})\n")

    path.write_text("\n".join(md), encoding="utf-8")
    return path
