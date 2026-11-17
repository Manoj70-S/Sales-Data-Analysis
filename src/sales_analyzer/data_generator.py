"""Generate realistic retail sales CSV data with seasonality, returns, discounts, and imperfections."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np


PRODUCTS = [
    ("Laptop Pro 15", "Electronics", 1299.0, 820.0, 18.0),
    ("Wireless Headphones", "Electronics", 199.0, 78.0, 6.0),
    ("4K Monitor 27\"", "Electronics", 449.0, 275.0, 14.0),
    ("Mechanical Keyboard", "Accessories", 139.0, 52.0, 5.5),
    ("Ergonomic Mouse", "Accessories", 79.0, 28.0, 4.0),
    ("USB-C Docking Station", "Accessories", 159.0, 68.0, 5.0),
    ("Standing Desk", "Furniture", 599.0, 395.0, 35.0),
    ("Mesh Office Chair", "Furniture", 349.0, 210.0, 28.0),
    ("Desk Lamp LED", "Home Office", 49.0, 26.0, 4.5),
    ("Webcam HD 1080p", "Electronics", 89.0, 48.0, 4.5),
]

REGIONS = ["North", "South", "East", "West"]

# Base monthly demand weights (Jan..Dec) with Q4 holiday surge and summer dip
SEASONALITY = np.array(
    [0.82, 0.85, 0.94, 0.98, 1.02, 0.96, 0.91, 0.99, 1.08, 1.14, 1.32, 1.48],
    dtype=np.float64,
)

REGION_FACTORS = {
    "North": 1.15,
    "South": 0.92,
    "East": 1.08,
    "West": 0.88,
}


def generate_sample_sales_csv(
    output_path: str | Path,
    year: int = 2025,
    seed: int = 42,
    inject_dirty_data: bool = True,
) -> Path:
    """Create a realistic 12-month multi-region sales dataset CSV.

    Includes seasonality, steady growth trend, regional demand variations,
    discounts, returns, shipping costs, and optional real-world data flaws
    (missing values, negative/corrupt entries, and sales anomalies).
    """
    rng = np.random.default_rng(seed)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Product base monthly volumes (wide spread to demonstrate Pareto 80/20)
    base_volumes = np.array([34, 95, 42, 78, 115, 60, 24, 36, 52, 48], dtype=np.float64)
    # Monthly growth factor per product (some growing, one declining to test underperformer alerts)
    product_trends = np.array([0.025, 0.03, 0.015, 0.02, 0.01, 0.018, 0.012, 0.008, -0.045, -0.02])

    fieldnames = [
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

    rows: list[dict[str, object]] = []

    for m_idx in range(12):
        month_str = f"{year}-{m_idx + 1:02d}"
        season_mult = SEASONALITY[m_idx]

        for p_idx, (name, category, price, cost, shipping) in enumerate(PRODUCTS):
            trend_mult = max(0.35, 1.0 + product_trends[p_idx] * m_idx)

            for region in REGIONS:
                reg_mult = REGION_FACTORS[region]
                expected_units = base_volumes[p_idx] * season_mult * trend_mult * reg_mult / len(REGIONS)
                units = int(max(1, rng.poisson(lam=max(1.0, expected_units))))

                # Promo discounts heavier in Nov/Dec and on some accessories
                base_disc = 0.14 if m_idx in (10, 11) else 0.05
                discount_pct = round(float(np.clip(rng.normal(base_disc, 0.03), 0.0, 0.35)), 4)

                # Returns typically 2% - 8% (higher for chairs/webcams)
                return_prob = 0.07 if name in ("Mesh Office Chair", "Webcam HD 1080p") else 0.03
                returned = int(rng.binomial(n=units, p=return_prob))

                # Slight regional/monthly cost variation
                actual_price = round(float(price * rng.uniform(0.98, 1.02)), 2)
                actual_cost = round(float(cost * rng.uniform(0.97, 1.03)), 2)

                rows.append(
                    {
                        "month": month_str,
                        "product": name,
                        "category": category,
                        "region": region,
                        "units_sold": units,
                        "returned_units": returned,
                        "unit_price": actual_price,
                        "unit_cost": actual_cost,
                        "discount_pct": discount_pct,
                        "shipping_per_unit": shipping,
                    }
                )

    if inject_dirty_data and len(rows) > 20:
        # 1. Missing values (empty cells -> NaN during loading)
        rows[5]["unit_price"] = ""
        rows[42]["discount_pct"] = ""
        rows[110]["units_sold"] = ""
        rows[205]["returned_units"] = ""

        # 2. Invalid negative unit_cost / price or returns > units_sold
        rows[88]["unit_price"] = -199.0
        rows[150]["returned_units"] = int(rows[150]["units_sold"]) + 10  # type: ignore[arg-type]

        # 3. Real-world sales anomaly: huge corporate bulk order in July (East, Laptop Pro 15)
        for row in rows:
            if row["month"] == f"{year}-07" and row["product"] == "Laptop Pro 15" and row["region"] == "East":
                row["units_sold"] = int(row["units_sold"]) * 6  # type: ignore[arg-type]
                row["discount_pct"] = 0.12
                break

        # 4. Duplicate row to verify deduplication
        rows.append(dict(rows[12]))

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_file
