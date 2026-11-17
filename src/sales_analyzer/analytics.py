"""Pure NumPy real-world sales analytics engine.

Demonstrates vectorized array operations, broadcasting, 3D axis aggregations,
boolean masking, multi-criteria sorting, cumulative Pareto (ABC) analysis,
moving averages, Z-score anomaly detection, linear forecasting, correlation
matrices, and what-if scenario simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .loader import SalesDataset


@dataclass
class OverallKPIs:
    total_units_sold: float
    total_returned_units: float
    net_units_sold: float
    return_rate_pct: float
    gross_revenue: float
    total_discounts: float
    refund_value: float
    net_revenue: float
    total_cogs: float
    total_shipping: float
    net_profit: float
    profit_margin_pct: float
    avg_selling_price: float
    cagr_monthly_pct: float


@dataclass
class MonthlyMetrics:
    months: list[str]
    net_units: np.ndarray
    gross_revenue: np.ndarray
    net_revenue: np.ndarray
    net_profit: np.ndarray
    profit_margin_pct: np.ndarray
    return_rate_pct: np.ndarray
    mom_revenue_growth_pct: np.ndarray
    mom_profit_growth_pct: np.ndarray
    revenue_ma3: np.ndarray
    best_month: str
    worst_month: str
    best_growth_month: str | None
    worst_growth_month: str | None


@dataclass
class ProductMetrics:
    products: list[str]
    categories: list[str]
    net_units: np.ndarray
    net_revenue: np.ndarray
    net_profit: np.ndarray
    profit_margin_pct: np.ndarray
    return_rate_pct: np.ndarray
    revenue_share_pct: np.ndarray
    cumulative_revenue_share_pct: np.ndarray
    abc_tier: list[str]
    revenue_trend_slope: np.ndarray
    rank_by_revenue: np.ndarray
    rank_by_profit: np.ndarray
    rank_by_units: np.ndarray
    rank_by_margin: np.ndarray
    at_risk_products: list[str]


@dataclass
class RegionalMetrics:
    regions: list[str]
    net_units: np.ndarray
    net_revenue: np.ndarray
    net_profit: np.ndarray
    profit_margin_pct: np.ndarray
    revenue_share_pct: np.ndarray
    product_region_profit_matrix: np.ndarray  # shape (P, R)


@dataclass
class AnomalyRecord:
    month: str
    product: str
    region: str
    net_revenue: float
    z_score: float
    direction: str  # "spike" or "drop"


@dataclass
class ForecastResult:
    future_months: list[str]
    predicted_revenue: np.ndarray
    predicted_profit: np.ndarray
    revenue_slope_per_month: float
    profit_slope_per_month: float


@dataclass
class CorrelationPair:
    product_a: str
    product_b: str
    correlation: float


@dataclass
class ScenarioResult:
    label: str
    price_change_pct: float
    cost_change_pct: float
    discount_delta_pct: float
    baseline_revenue: float
    scenario_revenue: float
    revenue_delta: float
    revenue_delta_pct: float
    baseline_profit: float
    scenario_profit: float
    profit_delta: float
    profit_delta_pct: float
    scenario_margin_pct: float


@dataclass
class AnalysisResult:
    overall: OverallKPIs
    monthly: MonthlyMetrics
    products: ProductMetrics
    regional: RegionalMetrics
    anomalies: list[AnomalyRecord]
    forecast: ForecastResult
    top_correlations: list[CorrelationPair]
    scenarios: list[ScenarioResult]


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Vectorized safe division avoiding ZeroDivisionError / inf / NaN."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), fill, dtype=np.float64)
    return np.divide(num, den, out=out, where=(den != 0))


def _moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Compute trailing moving average with NaN padding for leading entries."""
    n = values.shape[0]
    ma = np.full(n, np.nan, dtype=np.float64)
    if n < window or window <= 0:
        return ma
    kernel = np.ones(window, dtype=np.float64) / window
    ma[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return ma


def _mom_growth(series: np.ndarray) -> np.ndarray:
    """Month-over-Month percentage growth; index 0 is NaN, zero previous period -> NaN."""
    n = series.shape[0]
    growth = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return growth
    prev = series[:-1]
    curr = series[1:]
    valid = prev != 0
    growth[1:] = np.where(valid, ((curr - prev) / np.where(valid, prev, 1.0)) * 100.0, np.nan)
    return growth


def _next_month_labels(last_month: str, count: int) -> list[str]:
    """Generate YYYY-MM labels following last_month."""
    try:
        year_str, month_str = last_month.split("-")
        year, month = int(year_str), int(month_str)
    except ValueError:
        return [f"Forecast+{i + 1}" for i in range(count)]

    labels: list[str] = []
    for _ in range(count):
        month += 1
        if month > 12:
            month = 1
            year += 1
        labels.append(f"{year:04d}-{month:02d}")
    return labels


def compute_financial_tensors(
    dataset: SalesDataset,
    price_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    discount_delta: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute 3D financial tensors (M, P, R) using vectorized NumPy operations."""
    eff_price = dataset.unit_price * price_multiplier
    eff_cost = dataset.unit_cost * cost_multiplier
    eff_discount = np.clip(dataset.discount_pct + discount_delta, 0.0, 0.95)

    net_units = np.maximum(0.0, dataset.units_sold - dataset.returned_units)

    gross_revenue = dataset.units_sold * eff_price
    discount_value = gross_revenue * eff_discount
    billed_revenue = gross_revenue - discount_value
    refund_value = dataset.returned_units * eff_price * (1.0 - eff_discount)
    net_revenue = billed_revenue - refund_value  # == net_units * eff_price * (1 - eff_discount)

    # COGS applies to net kept units; shipping is incurred on all shipped units
    cogs = net_units * eff_cost
    shipping = dataset.units_sold * dataset.shipping_per_unit
    net_profit = net_revenue - cogs - shipping

    return {
        "net_units": net_units,
        "gross_revenue": gross_revenue,
        "discount_value": discount_value,
        "refund_value": refund_value,
        "net_revenue": net_revenue,
        "cogs": cogs,
        "shipping": shipping,
        "net_profit": net_profit,
    }


def simulate_what_if(
    dataset: SalesDataset,
    label: str,
    price_change_pct: float = 0.0,
    cost_change_pct: float = 0.0,
    discount_delta_pct: float = 0.0,
) -> ScenarioResult:
    """Run a vectorized what-if financial simulation against baseline."""
    base = compute_financial_tensors(dataset)
    sim = compute_financial_tensors(
        dataset,
        price_multiplier=1.0 + price_change_pct / 100.0,
        cost_multiplier=1.0 + cost_change_pct / 100.0,
        discount_delta=discount_delta_pct / 100.0,
    )

    base_rev = float(np.sum(base["net_revenue"]))
    sim_rev = float(np.sum(sim["net_revenue"]))
    base_prof = float(np.sum(base["net_profit"]))
    sim_prof = float(np.sum(sim["net_profit"]))

    rev_delta = sim_rev - base_rev
    prof_delta = sim_prof - base_prof

    rev_delta_pct = float(_safe_divide(rev_delta, base_rev) * 100.0)
    prof_delta_pct = float(_safe_divide(prof_delta, abs(base_prof)) * 100.0)
    sim_margin_pct = float(_safe_divide(sim_prof, sim_rev) * 100.0)

    return ScenarioResult(
        label=label,
        price_change_pct=price_change_pct,
        cost_change_pct=cost_change_pct,
        discount_delta_pct=discount_delta_pct,
        baseline_revenue=base_rev,
        scenario_revenue=sim_rev,
        revenue_delta=rev_delta,
        revenue_delta_pct=rev_delta_pct,
        baseline_profit=base_prof,
        scenario_profit=sim_prof,
        profit_delta=prof_delta,
        profit_delta_pct=prof_delta_pct,
        scenario_margin_pct=sim_margin_pct,
    )


def analyze_sales(
    dataset: SalesDataset,
    forecast_months: int = 3,
    anomaly_z_threshold: float = 2.5,
    custom_scenario: tuple[float, float, float] | None = None,
) -> AnalysisResult:
    """Run the complete NumPy sales analytics pipeline."""
    tensors = compute_financial_tensors(dataset)

    units_sold = dataset.units_sold
    returned_units = dataset.returned_units
    net_units = tensors["net_units"]
    gross_revenue = tensors["gross_revenue"]
    discount_value = tensors["discount_value"]
    refund_value = tensors["refund_value"]
    net_revenue = tensors["net_revenue"]
    cogs = tensors["cogs"]
    shipping = tensors["shipping"]
    net_profit = tensors["net_profit"]

    # 1. Monthly aggregations along axes (1=products, 2=regions)
    m_units_sold = np.sum(units_sold, axis=(1, 2))
    m_returned = np.sum(returned_units, axis=(1, 2))
    m_net_units = np.sum(net_units, axis=(1, 2))
    m_gross_rev = np.sum(gross_revenue, axis=(1, 2))
    m_net_rev = np.sum(net_revenue, axis=(1, 2))
    m_net_profit = np.sum(net_profit, axis=(1, 2))
    m_margin_pct = _safe_divide(m_net_profit, m_net_rev) * 100.0
    m_return_rate_pct = _safe_divide(m_returned, m_units_sold) * 100.0

    mom_rev_growth = _mom_growth(m_net_rev)
    mom_profit_growth = _mom_growth(m_net_profit)
    rev_ma3 = _moving_average(m_net_rev, window=3)

    best_month = dataset.months[int(np.argmax(m_net_rev))]
    worst_month = dataset.months[int(np.argmin(m_net_rev))]

    valid_growth = ~np.isnan(mom_rev_growth)
    if np.any(valid_growth):
        best_growth_idx = int(np.nanargmax(mom_rev_growth))
        worst_growth_idx = int(np.nanargmin(mom_rev_growth))
        best_growth_month = dataset.months[best_growth_idx]
        worst_growth_month = dataset.months[worst_growth_idx]
    else:
        best_growth_month = None
        worst_growth_month = None

    monthly_metrics = MonthlyMetrics(
        months=dataset.months,
        net_units=m_net_units,
        gross_revenue=m_gross_rev,
        net_revenue=m_net_rev,
        net_profit=m_net_profit,
        profit_margin_pct=m_margin_pct,
        return_rate_pct=m_return_rate_pct,
        mom_revenue_growth_pct=mom_rev_growth,
        mom_profit_growth_pct=mom_profit_growth,
        revenue_ma3=rev_ma3,
        best_month=best_month,
        worst_month=worst_month,
        best_growth_month=best_growth_month,
        worst_growth_month=worst_growth_month,
    )

    # 2. Overall KPIs
    tot_units = float(np.sum(units_sold))
    tot_returned = float(np.sum(returned_units))
    tot_net_units = float(np.sum(net_units))
    tot_gross_rev = float(np.sum(gross_revenue))
    tot_discounts = float(np.sum(discount_value))
    tot_refunds = float(np.sum(refund_value))
    tot_net_rev = float(np.sum(net_revenue))
    tot_cogs = float(np.sum(cogs))
    tot_shipping = float(np.sum(shipping))
    tot_net_profit = float(np.sum(net_profit))

    # Compound Monthly Growth Rate (CMGR)
    if len(m_net_rev) >= 2 and m_net_rev[0] > 0 and m_net_rev[-1] > 0:
        cmgr_pct = float(((m_net_rev[-1] / m_net_rev[0]) ** (1.0 / (len(m_net_rev) - 1)) - 1.0) * 100.0)
    else:
        cmgr_pct = 0.0

    overall_kpis = OverallKPIs(
        total_units_sold=tot_units,
        total_returned_units=tot_returned,
        net_units_sold=tot_net_units,
        return_rate_pct=float(_safe_divide(tot_returned, tot_units) * 100.0),
        gross_revenue=tot_gross_rev,
        total_discounts=tot_discounts,
        refund_value=tot_refunds,
        net_revenue=tot_net_rev,
        total_cogs=tot_cogs,
        total_shipping=tot_shipping,
        net_profit=tot_net_profit,
        profit_margin_pct=float(_safe_divide(tot_net_profit, tot_net_rev) * 100.0),
        avg_selling_price=float(_safe_divide(tot_net_rev, tot_net_units)),
        cagr_monthly_pct=cmgr_pct,
    )

    # 3. Product aggregations along axes (0=months, 2=regions)
    p_units_sold = np.sum(units_sold, axis=(0, 2))
    p_returned = np.sum(returned_units, axis=(0, 2))
    p_net_units = np.sum(net_units, axis=(0, 2))
    p_net_rev = np.sum(net_revenue, axis=(0, 2))
    p_net_profit = np.sum(net_profit, axis=(0, 2))
    p_margin_pct = _safe_divide(p_net_profit, p_net_rev) * 100.0
    p_return_rate_pct = _safe_divide(p_returned, p_units_sold) * 100.0
    p_rev_share = _safe_divide(p_net_rev, tot_net_rev) * 100.0

    # Multi-criteria rankings (descending)
    rank_by_rev = np.argsort(p_net_rev)[::-1]
    rank_by_profit = np.argsort(p_net_profit)[::-1]
    rank_by_units = np.argsort(p_net_units)[::-1]
    rank_by_margin = np.argsort(p_margin_pct)[::-1]

    # ABC / Pareto classification (80/15/5 rule) computed in revenue-descending order
    sorted_shares = p_rev_share[rank_by_rev]
    sorted_cum_shares = np.cumsum(sorted_shares)
    cum_shares_in_orig_order = np.zeros_like(p_rev_share)
    cum_shares_in_orig_order[rank_by_rev] = sorted_cum_shares

    abc_tier = ["C"] * len(dataset.products)
    for pos, p_idx in enumerate(rank_by_rev):
        prev_cum = sorted_cum_shares[pos - 1] if pos > 0 else 0.0
        if prev_cum < 80.0:
            abc_tier[int(p_idx)] = "A"
        elif prev_cum < 95.0:
            abc_tier[int(p_idx)] = "B"
        else:
            abc_tier[int(p_idx)] = "C"

    # Per-product monthly revenue trend slope using np.polyfit
    monthly_prod_rev = np.sum(net_revenue, axis=2)  # shape (M, P)
    num_months = len(dataset.months)
    x_months = np.arange(num_months, dtype=np.float64)
    p_slopes = np.zeros(len(dataset.products), dtype=np.float64)
    if num_months >= 2:
        for p_idx in range(len(dataset.products)):
            slope, _ = np.polyfit(x_months, monthly_prod_rev[:, p_idx], deg=1)
            p_slopes[p_idx] = float(slope)

    # Flag at-risk products: declining trend OR low profit margin (<20%) OR high return rate (>6%)
    at_risk_mask = (p_slopes < 0) | (p_margin_pct < 20.0) | (p_return_rate_pct > 6.0)
    at_risk_products = [dataset.products[i] for i in np.where(at_risk_mask)[0]]

    product_metrics = ProductMetrics(
        products=dataset.products,
        categories=[dataset.product_categories[p] for p in dataset.products],
        net_units=p_net_units,
        net_revenue=p_net_rev,
        net_profit=p_net_profit,
        profit_margin_pct=p_margin_pct,
        return_rate_pct=p_return_rate_pct,
        revenue_share_pct=p_rev_share,
        cumulative_revenue_share_pct=cum_shares_in_orig_order,
        abc_tier=abc_tier,
        revenue_trend_slope=p_slopes,
        rank_by_revenue=rank_by_rev,
        rank_by_profit=rank_by_profit,
        rank_by_units=rank_by_units,
        rank_by_margin=rank_by_margin,
        at_risk_products=at_risk_products,
    )

    # 4. Regional aggregations along axes (0=months, 1=products)
    r_net_units = np.sum(net_units, axis=(0, 1))
    r_net_rev = np.sum(net_revenue, axis=(0, 1))
    r_net_profit = np.sum(net_profit, axis=(0, 1))
    r_margin_pct = _safe_divide(r_net_profit, r_net_rev) * 100.0
    r_rev_share = _safe_divide(r_net_rev, tot_net_rev) * 100.0
    prod_region_profit = np.sum(net_profit, axis=0)  # shape (P, R)

    regional_metrics = RegionalMetrics(
        regions=dataset.regions,
        net_units=r_net_units,
        net_revenue=r_net_rev,
        net_profit=r_net_profit,
        profit_margin_pct=r_margin_pct,
        revenue_share_pct=r_rev_share,
        product_region_profit_matrix=prod_region_profit,
    )

    # 5. Anomaly Detection (per-product Z-score across months & regions)
    anomalies: list[AnomalyRecord] = []
    # Compute mean and std per product across (months, regions), keeping dims for broadcasting
    prod_means = np.mean(net_revenue, axis=(0, 2), keepdims=True)
    prod_stds = np.std(net_revenue, axis=(0, 2), keepdims=True)
    z_scores = _safe_divide(net_revenue - prod_means, prod_stds, fill=0.0)

    anomaly_indices = np.argwhere(np.abs(z_scores) >= anomaly_z_threshold)
    for mi, pi, ri in anomaly_indices:
        z_val = float(z_scores[mi, pi, ri])
        anomalies.append(
            AnomalyRecord(
                month=dataset.months[int(mi)],
                product=dataset.products[int(pi)],
                region=dataset.regions[int(ri)],
                net_revenue=float(net_revenue[mi, pi, ri]),
                z_score=z_val,
                direction="spike" if z_val > 0 else "drop",
            )
        )
    anomalies.sort(key=lambda a: abs(a.z_score), reverse=True)

    # 6. Linear Trend Forecasting (next N months via np.polyfit)
    future_labels = _next_month_labels(dataset.months[-1], forecast_months)
    if num_months >= 2 and forecast_months > 0:
        rev_coeffs = np.polyfit(x_months, m_net_rev, deg=1)
        prof_coeffs = np.polyfit(x_months, m_net_profit, deg=1)
        future_x = np.arange(num_months, num_months + forecast_months, dtype=np.float64)
        pred_rev = np.maximum(0.0, np.polyval(rev_coeffs, future_x))
        pred_prof = np.polyval(prof_coeffs, future_x)
        rev_slope = float(rev_coeffs[0])
        prof_slope = float(prof_coeffs[0])
    else:
        pred_rev = np.full(forecast_months, m_net_rev[-1] if num_months > 0 else 0.0, dtype=np.float64)
        pred_prof = np.full(forecast_months, m_net_profit[-1] if num_months > 0 else 0.0, dtype=np.float64)
        rev_slope = 0.0
        prof_slope = 0.0

    forecast = ForecastResult(
        future_months=future_labels,
        predicted_revenue=pred_rev,
        predicted_profit=pred_prof,
        revenue_slope_per_month=rev_slope,
        profit_slope_per_month=prof_slope,
    )

    # 7. Product Co-Movement / Correlation Matrix (across months)
    top_correlations: list[CorrelationPair] = []
    if len(dataset.products) >= 2 and num_months >= 3:
        # Transpose to shape (P, M) for np.corrcoef
        prod_series = monthly_prod_rev.T
        valid_std = np.std(prod_series, axis=1) > 0
        if np.sum(valid_std) >= 2:
            corr_matrix = np.corrcoef(prod_series)
            pairs: list[CorrelationPair] = []
            num_p = len(dataset.products)
            for i in range(num_p):
                for j in range(i + 1, num_p):
                    c_val = corr_matrix[i, j]
                    if not np.isnan(c_val):
                        pairs.append(
                            CorrelationPair(
                                product_a=dataset.products[i],
                                product_b=dataset.products[j],
                                correlation=float(c_val),
                            )
                        )
            pairs.sort(key=lambda pair: abs(pair.correlation), reverse=True)
            top_correlations = pairs[:5]

    # 8. Built-in What-If Scenario Simulations (+ optional user-supplied scenario)
    scenarios = [
        simulate_what_if(dataset, "Price +5% (Hold Volume)", price_change_pct=5.0),
        simulate_what_if(dataset, "Supplier Cost +8% Inflation", cost_change_pct=8.0),
        simulate_what_if(dataset, "Promo Push (+5% Extra Discount)", discount_delta_pct=5.0),
        simulate_what_if(dataset, "Margin Optimization (+3% Price, -2% Discount)", price_change_pct=3.0, discount_delta_pct=-2.0),
    ]
    if custom_scenario is not None:
        p_chg, c_chg, d_chg = custom_scenario
        scenarios.append(
            simulate_what_if(
                dataset,
                f"Custom (Price {p_chg:+.1f}%, Cost {c_chg:+.1f}%, Disc {d_chg:+.1f}%)",
                price_change_pct=p_chg,
                cost_change_pct=c_chg,
                discount_delta_pct=d_chg,
            )
        )

    return AnalysisResult(
        overall=overall_kpis,
        monthly=monthly_metrics,
        products=product_metrics,
        regional=regional_metrics,
        anomalies=anomalies,
        forecast=forecast,
        top_correlations=top_correlations,
        scenarios=scenarios,
    )
