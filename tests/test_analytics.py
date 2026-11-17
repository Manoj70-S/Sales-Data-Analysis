"""Unit tests for Sales Data Analyzer loader, cleaning, and NumPy analytics engine."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sales_analyzer.analytics import analyze_sales, simulate_what_if
from sales_analyzer.data_generator import generate_sample_sales_csv
from sales_analyzer.loader import CleaningReport, SalesDataset, load_sales_csv


def _make_simple_dataset(
    units: np.ndarray,
    returned: np.ndarray | None = None,
    price: float = 100.0,
    cost: float = 50.0,
    discount: float = 0.0,
    shipping: float = 0.0,
) -> SalesDataset:
    shape = units.shape
    m, p, r = shape
    return SalesDataset(
        months=[f"2025-{i + 1:02d}" for i in range(m)],
        products=[f"Prod{i + 1}" for i in range(p)],
        product_categories={f"Prod{i + 1}": "CatA" for i in range(p)},
        regions=[f"Reg{i + 1}" for i in range(r)],
        units_sold=units.astype(np.float64),
        returned_units=np.zeros(shape, dtype=np.float64) if returned is None else returned.astype(np.float64),
        unit_price=np.full(shape, price, dtype=np.float64),
        unit_cost=np.full(shape, cost, dtype=np.float64),
        discount_pct=np.full(shape, discount, dtype=np.float64),
        shipping_per_unit=np.full(shape, shipping, dtype=np.float64),
        cleaning_report=CleaningReport(),
    )


def test_revenue_profit_and_returns_math():
    # 2 months, 1 product, 1 region
    # Month 1: 10 sold, 2 returned, price=100, discount=10%, cost=40, shipping=5
    # Gross = 1000, Discount = 100, Refund = 2 * 90 = 180 -> Net Rev = 720
    # COGS = 8 * 40 = 320, Shipping = 10 * 5 = 50 -> Net Profit = 720 - 320 - 50 = 350
    units = np.array([[[10.0]], [[20.0]]])
    returned = np.array([[[2.0]], [[0.0]]])
    ds = _make_simple_dataset(units, returned=returned, price=100.0, cost=40.0, discount=0.10, shipping=5.0)

    res = analyze_sales(ds, forecast_months=2)

    assert np.isclose(res.monthly.net_revenue[0], 720.0)
    assert np.isclose(res.monthly.net_profit[0], 350.0)
    # Month 2: 20 sold, 0 returned -> Net Rev = 1800, COGS = 800, Ship = 100 -> Net Profit = 900
    assert np.isclose(res.monthly.net_revenue[1], 1800.0)
    assert np.isclose(res.monthly.net_profit[1], 900.0)
    # MoM revenue growth: (1800 - 720) / 720 * 100 = +150%
    assert np.isnan(res.monthly.mom_revenue_growth_pct[0])
    assert np.isclose(res.monthly.mom_revenue_growth_pct[1], 150.0)


def test_zero_revenue_month_mom_growth_is_nan():
    units = np.array([[[0.0]], [[10.0]], [[20.0]]])
    ds = _make_simple_dataset(units)
    res = analyze_sales(ds)

    assert np.isnan(res.monthly.mom_revenue_growth_pct[0])
    assert np.isnan(res.monthly.mom_revenue_growth_pct[1])  # division by 0 safely handled
    assert np.isclose(res.monthly.mom_revenue_growth_pct[2], 100.0)


def test_pareto_abc_classification_and_sorting():
    # 1 month, 4 products, 1 region: revenues 820, 110, 50, 20 (total 1000)
    # Shares: 82% (A), 11% (cum 93% -> prev 82% => B), 5% (cum 98% -> prev 93% => B), 2% (prev 98% => C)
    units = np.array([[[2.0], [82.0], [5.0], [11.0]]])
    ds = _make_simple_dataset(units, price=10.0, cost=5.0)
    res = analyze_sales(ds)

    assert res.products.products[int(res.products.rank_by_revenue[0])] == "Prod2"
    assert res.products.abc_tier[1] == "A"
    assert res.products.abc_tier[3] == "B"
    assert res.products.abc_tier[2] == "B"
    assert res.products.abc_tier[0] == "C"


def test_anomaly_detection_flags_outlier():
    # 10 months of stable sales (~10 units) with one huge spike (100 units) in month 5
    units = np.full((10, 1, 2), 10.0)
    units[4, 0, 1] = 100.0
    ds = _make_simple_dataset(units, price=100.0, cost=50.0)
    res = analyze_sales(ds, anomaly_z_threshold=2.5)

    assert len(res.anomalies) == 1
    assert res.anomalies[0].month == "2025-05"
    assert res.anomalies[0].region == "Reg2"
    assert res.anomalies[0].direction == "spike"
    assert res.anomalies[0].z_score > 2.5


def test_linear_forecast_exact_trend():
    # Perfectly linear monthly net revenue: 100, 200, 300, 400 -> next 2 should be 500, 600
    units = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    ds = _make_simple_dataset(units, price=100.0, cost=60.0)
    res = analyze_sales(ds, forecast_months=2)

    assert res.forecast.future_months == ["2025-05", "2025-06"]
    assert np.allclose(res.forecast.predicted_revenue, [500.0, 600.0])
    assert np.allclose(res.forecast.predicted_profit, [200.0, 240.0])


def test_what_if_scenario_modeling():
    units = np.array([[[10.0]], [[10.0]]])
    ds = _make_simple_dataset(units, price=100.0, cost=50.0)
    # Baseline rev = 2000, profit = 1000
    # +10% price -> rev = 2200, profit = 1200 (+200, +20%)
    scenario = simulate_what_if(ds, "Price +10%", price_change_pct=10.0)
    assert np.isclose(scenario.scenario_revenue, 2200.0)
    assert np.isclose(scenario.scenario_profit, 1200.0)
    assert np.isclose(scenario.profit_delta_pct, 20.0)


def test_csv_loader_cleans_dirty_data(tmp_path: Path):
    csv_file = tmp_path / "dirty.csv"
    rows = [
        # Normal row
        ["2025-01", "Widget", "Gadgets", "North", "10", "1", "100", "50", "0.10", "5"],
        # Exact duplicate of row 1
        ["2025-01", "Widget", "Gadgets", "North", "10", "1", "100", "50", "0.10", "5"],
        # Missing unit_price + negative cost + returns > units
        ["2025-02", "Widget", "Gadgets", "North", "8", "15", "", "-50", "0.10", "5"],
    ]
    header = [
        "month",
        "product",
        "category",
        "region",
        "units_sold",
        "returned_units",
        "unit_price",
        "unit_cost",
        "discount_pct",
        "shipping_per_unit",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    ds = load_sales_csv(csv_file)

    assert ds.cleaning_report.duplicate_rows_removed == 1
    assert ds.cleaning_report.missing_values_imputed["unit_price"] == 1
    assert ds.cleaning_report.invalid_values_fixed["negative_unit_cost"] == 1
    assert ds.cleaning_report.invalid_values_fixed["returns_exceeding_units"] == 1
    assert np.isclose(ds.unit_price[1, 0, 0], 100.0)  # imputed from product median
    assert np.isclose(ds.unit_cost[1, 0, 0], 50.0)  # abs(-50)
    assert np.isclose(ds.returned_units[1, 0, 0], 8.0)  # clipped to units_sold


def test_end_to_end_sample_generation_and_analysis(tmp_path: Path):
    sample_csv = generate_sample_sales_csv(tmp_path / "sample.csv", inject_dirty_data=True)
    ds = load_sales_csv(sample_csv)
    res = analyze_sales(ds, forecast_months=3)

    assert ds.shape == (12, 10, 4)
    assert ds.cleaning_report.total_issues_fixed >= 6
    assert res.overall.net_revenue > 0
    assert res.overall.net_profit > 0
    assert len(res.anomalies) >= 1  # Should catch the injected July East bulk order spike
    assert len(res.forecast.future_months) == 3
