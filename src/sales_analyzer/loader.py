"""CSV loader, schema validator, data cleaner, and 3D NumPy tensor builder."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


REQUIRED_COLUMNS = {
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
}


@dataclass
class CleaningReport:
    """Audit trail of data quality issues found and fixed during ingestion."""

    total_rows_read: int = 0
    duplicate_rows_removed: int = 0
    missing_values_imputed: dict[str, int] = field(default_factory=dict)
    invalid_values_fixed: dict[str, int] = field(default_factory=dict)
    filtered_out_rows: int = 0

    @property
    def total_issues_fixed(self) -> int:
        return (
            self.duplicate_rows_removed
            + sum(self.missing_values_imputed.values())
            + sum(self.invalid_values_fixed.values())
        )


@dataclass
class SalesDataset:
    """Structured 3D NumPy sales dataset with shape (months, products, regions)."""

    months: list[str]
    products: list[str]
    product_categories: dict[str, str]
    regions: list[str]

    # 3D float64 tensors of shape (M, P, R)
    units_sold: np.ndarray
    returned_units: np.ndarray
    unit_price: np.ndarray
    unit_cost: np.ndarray
    discount_pct: np.ndarray
    shipping_per_unit: np.ndarray

    cleaning_report: CleaningReport

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.months), len(self.products), len(self.regions))


def _parse_float(value: str | None) -> float:
    if value is None:
        return np.nan
    stripped = value.strip()
    if stripped == "" or stripped.lower() in {"nan", "null", "none", "n/a"}:
        return np.nan
    try:
        return float(stripped)
    except ValueError:
        return np.nan


def _impute_along_product_median(tensor: np.ndarray, default_value: float = 0.0) -> tuple[np.ndarray, int]:
    """Replace NaNs with the product's median across months/regions (fallback: global median or default)."""
    nan_mask = np.isnan(tensor)
    missing_count = int(np.sum(nan_mask))
    if missing_count == 0:
        return tensor, 0

    cleaned = tensor.copy()
    valid_global = cleaned[~nan_mask]
    global_fallback = float(np.median(valid_global)) if valid_global.size > 0 else default_value

    # Compute per-product median across months (axis=0) and regions (axis=2)
    num_products = cleaned.shape[1]
    for p_idx in range(num_products):
        prod_slice = cleaned[:, p_idx, :]
        prod_nans = np.isnan(prod_slice)
        if np.any(prod_nans):
            valid_prod = prod_slice[~prod_nans]
            fill_val = float(np.median(valid_prod)) if valid_prod.size > 0 else global_fallback
            prod_slice[prod_nans] = fill_val
            cleaned[:, p_idx, :] = prod_slice

    return cleaned, missing_count


def load_sales_csv(
    csv_path: str | Path,
    region_filter: str | None = None,
    category_filter: str | None = None,
) -> SalesDataset:
    """Load, validate, clean, and reshape a sales CSV into 3D NumPy arrays."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Sales CSV file not found: {path}")

    report = CleaningReport()

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or missing a header row.")

        header_set = {col.strip() for col in reader.fieldnames}
        missing_cols = REQUIRED_COLUMNS - header_set
        if missing_cols:
            raise ValueError(f"Missing required CSV columns: {sorted(missing_cols)}")

        raw_rows: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str, str, str, str]] = set()

        for row in reader:
            report.total_rows_read += 1

            # Optional region / category filters
            row_region = row["region"].strip()
            row_category = row["category"].strip()
            if region_filter and row_region.lower() != region_filter.strip().lower():
                report.filtered_out_rows += 1
                continue
            if category_filter and row_category.lower() != category_filter.strip().lower():
                report.filtered_out_rows += 1
                continue

            # Exact duplicate check on (month, product, region, units_sold, unit_price)
            dedup_key = (
                row["month"].strip(),
                row["product"].strip(),
                row_region,
                row["units_sold"].strip(),
                row["unit_price"].strip(),
            )
            if dedup_key in seen_keys:
                report.duplicate_rows_removed += 1
                continue
            seen_keys.add(dedup_key)
            raw_rows.append(row)

    if not raw_rows:
        raise ValueError("No sales records available after loading/filtering.")

    months = sorted({r["month"].strip() for r in raw_rows})
    products = sorted({r["product"].strip() for r in raw_rows})
    regions = sorted({r["region"].strip() for r in raw_rows})

    product_categories: dict[str, str] = {}
    for r in raw_rows:
        product_categories[r["product"].strip()] = r["category"].strip()

    month_idx = {m: i for i, m in enumerate(months)}
    prod_idx = {p: i for i, p in enumerate(products)}
    reg_idx = {reg: i for i, reg in enumerate(regions)}

    shape = (len(months), len(products), len(regions))

    # Track whether a cell was explicitly present in the CSV vs genuinely missing data
    present_mask = np.zeros(shape, dtype=bool)
    units_sold = np.full(shape, np.nan, dtype=np.float64)
    returned_units = np.full(shape, np.nan, dtype=np.float64)
    unit_price = np.full(shape, np.nan, dtype=np.float64)
    unit_cost = np.full(shape, np.nan, dtype=np.float64)
    discount_pct = np.full(shape, np.nan, dtype=np.float64)
    shipping_per_unit = np.full(shape, np.nan, dtype=np.float64)

    # Aggregate if multiple non-duplicate rows map to the same (month, product, region)
    counts = np.zeros(shape, dtype=np.int64)

    for r in raw_rows:
        mi = month_idx[r["month"].strip()]
        pi = prod_idx[r["product"].strip()]
        ri = reg_idx[r["region"].strip()]

        u = _parse_float(r.get("units_sold"))
        ret = _parse_float(r.get("returned_units"))
        price = _parse_float(r.get("unit_price"))
        cost = _parse_float(r.get("unit_cost"))
        disc = _parse_float(r.get("discount_pct"))
        ship = _parse_float(r.get("shipping_per_unit"))

        if not present_mask[mi, pi, ri]:
            present_mask[mi, pi, ri] = True
            units_sold[mi, pi, ri] = u
            returned_units[mi, pi, ri] = ret
            unit_price[mi, pi, ri] = price
            unit_cost[mi, pi, ri] = cost
            discount_pct[mi, pi, ri] = disc
            shipping_per_unit[mi, pi, ri] = ship
            counts[mi, pi, ri] = 1
        else:
            # Sum quantities; running average for prices/rates
            c = counts[mi, pi, ri]
            units_sold[mi, pi, ri] = np.nansum([units_sold[mi, pi, ri], u])
            returned_units[mi, pi, ri] = np.nansum([returned_units[mi, pi, ri], ret])
            if not np.isnan(price):
                unit_price[mi, pi, ri] = price if np.isnan(unit_price[mi, pi, ri]) else (unit_price[mi, pi, ri] * c + price) / (c + 1)
            if not np.isnan(cost):
                unit_cost[mi, pi, ri] = cost if np.isnan(unit_cost[mi, pi, ri]) else (unit_cost[mi, pi, ri] * c + cost) / (c + 1)
            if not np.isnan(disc):
                discount_pct[mi, pi, ri] = disc if np.isnan(discount_pct[mi, pi, ri]) else (discount_pct[mi, pi, ri] * c + disc) / (c + 1)
            if not np.isnan(ship):
                shipping_per_unit[mi, pi, ri] = ship if np.isnan(shipping_per_unit[mi, pi, ri]) else (shipping_per_unit[mi, pi, ri] * c + ship) / (c + 1)
            counts[mi, pi, ri] = c + 1

    # Unobserved (month, product, region) combinations mean 0 units sold, not missing data
    unobserved = ~present_mask
    units_sold[unobserved] = 0.0
    returned_units[unobserved] = 0.0

    # Fix invalid negative prices/costs/units BEFORE imputation so medians aren't skewed
    neg_price_mask = ~np.isnan(unit_price) & (unit_price < 0)
    if np.any(neg_price_mask):
        report.invalid_values_fixed["negative_unit_price"] = int(np.sum(neg_price_mask))
        unit_price[neg_price_mask] = np.abs(unit_price[neg_price_mask])

    neg_cost_mask = ~np.isnan(unit_cost) & (unit_cost < 0)
    if np.any(neg_cost_mask):
        report.invalid_values_fixed["negative_unit_cost"] = int(np.sum(neg_cost_mask))
        unit_cost[neg_cost_mask] = np.abs(unit_cost[neg_cost_mask])

    neg_units_mask = ~np.isnan(units_sold) & (units_sold < 0)
    if np.any(neg_units_mask):
        report.invalid_values_fixed["negative_units_sold"] = int(np.sum(neg_units_mask))
        units_sold[neg_units_mask] = 0.0

    # Impute missing values along product median
    for field_name, arr, default_val in [
        ("units_sold", units_sold, 0.0),
        ("returned_units", returned_units, 0.0),
        ("unit_price", unit_price, 0.0),
        ("unit_cost", unit_cost, 0.0),
        ("discount_pct", discount_pct, 0.0),
        ("shipping_per_unit", shipping_per_unit, 0.0),
    ]:
        cleaned, missing_cnt = _impute_along_product_median(arr, default_value=default_val)
        # Only count missing values that occurred in actually observed rows for price/cost/discount/shipping
        observed_missing = int(np.sum(np.isnan(arr) & present_mask))
        if observed_missing > 0:
            report.missing_values_imputed[field_name] = observed_missing
        arr[:] = cleaned

    # Clamp discount_pct to [0.0, 0.95]
    bad_disc = (discount_pct < 0.0) | (discount_pct > 0.95)
    if np.any(bad_disc):
        report.invalid_values_fixed["out_of_range_discount"] = int(np.sum(bad_disc))
        discount_pct = np.clip(discount_pct, 0.0, 0.95)

    # Ensure returned_units <= units_sold and >= 0
    bad_returns = (returned_units < 0) | (returned_units > units_sold)
    if np.any(bad_returns):
        report.invalid_values_fixed["returns_exceeding_units"] = int(np.sum(bad_returns))
        returned_units = np.clip(returned_units, 0.0, units_sold)

    return SalesDataset(
        months=months,
        products=products,
        product_categories=product_categories,
        regions=regions,
        units_sold=units_sold,
        returned_units=returned_units,
        unit_price=unit_price,
        unit_cost=unit_cost,
        discount_pct=discount_pct,
        shipping_per_unit=shipping_per_unit,
        cleaning_report=report,
    )
