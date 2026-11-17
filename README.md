# Sales Data Analyzer (NumPy)

A real-world, production-structured **Sales Data Analyzer** built around **NumPy** for fast vectorized analytics, data cleaning, multi-dimensional KPI computation, anomaly detection, forecasting, what-if simulations, and executive reporting.

## Features

1. **Real-World CSV Ingestion & Cleaning (`src/sales_analyzer/loader.py`)**
   - Schema validation & exact duplicate removal
   - Automatic imputation of missing (`NaN`) values using per-product medians
   - Sanitization of invalid negative prices/costs, out-of-range discounts, and returns exceeding units sold
   - Reshapes tabular sales records into 3D NumPy tensors of shape `(months, products, regions)`

2. **Core NumPy Analytics Engine (`src/sales_analyzer/analytics.py`)**
   - **Financial Breakdown:** Gross Revenue, Discounts, Refunds/Returns, Net Revenue, COGS, Shipping, Net Profit, Profit Margin %, Return Rate %, and Average Selling Price
   - **Axis Aggregations:** Monthly (`axis=(1, 2)`), Product (`axis=(0, 2)`), Regional (`axis=(0, 1)`), and Product × Region Profit Matrix (`axis=0`)
   - **Growth & Trends:** Safe Month-over-Month (MoM) growth %, Compound Monthly Growth Rate (CMGR), and 3-month trailing moving averages (`np.convolve`)
   - **Product Intelligence:** Multi-criteria sorting (`np.argsort`), ABC / Pareto 80-20 classification (`np.cumsum`), per-product trend slopes, and automatic **At-Risk Watchlist** alerts
   - **Anomaly Detection:** Per-product Z-score outlier detection across months and regions
   - **Forecasting:** Linear trend projection (`np.polyfit` / `np.polyval`) for upcoming months
   - **Product Correlation:** Monthly revenue co-movement pairs via `np.corrcoef`
   - **What-If Modeling:** Vectorized price, supplier cost, and discount scenario simulations

3. **Executive Reporting & Visualization (`visualizer.py`, `reporter.py`)**
   - Terminal executive summary
   - Automated Markdown report (`output/sales_report.md`)
   - Exported CSV tables (`monthly_summary.csv`, `product_summary.csv`, `regional_summary.csv`)
   - 4 Matplotlib charts (`monthly_revenue_profit.png`, `monthly_growth.png`, `product_pareto.png`, `regional_profit_heatmap.png`)

## Quick Start

```powershell
# 1. Create and activate environment (or call .venv directly)
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt

# 2. Run analyzer (auto-generates a realistic sample dataset on first run)
.\.venv\Scripts\python.exe main.py

# 3. Filter by region or run a custom what-if scenario
.\.venv\Scripts\python.exe main.py --region East --what-if-price 4 --what-if-cost 6 --what-if-discount -2

# 4. Run unit tests
.\.venv\Scripts\pytest.exe -v
```

## Project Structure

```text
Sales Data Analyzer/
├── main.py                              # CLI entry point
├── requirements.txt                     # Pinned dependencies
├── src/
│   └── sales_analyzer/
│       ├── data_generator.py            # Realistic sample CSV generator (with seasonality & dirty data)
│       ├── loader.py                    # Schema validation, cleaning & 3D NumPy tensor builder
│       ├── analytics.py                 # Pure NumPy analytics, Pareto ABC, anomalies, forecasting & what-if
│       ├── visualizer.py                # Matplotlib chart generator
│       ├── reporter.py                  # Console, CSV & Markdown executive report exports
│       └── cli.py                       # argparse CLI
└── tests/
    └── test_analytics.py                # Pytest test suite
```
