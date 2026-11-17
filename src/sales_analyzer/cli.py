"""Command-line interface for the Sales Data Analyzer."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analytics import analyze_sales
from .data_generator import generate_sample_sales_csv
from .loader import load_sales_csv
from .reporter import export_csv_summaries, export_markdown_report, format_console_summary
from .visualizer import generate_charts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sales-analyzer",
        description="Real-world NumPy Sales Data Analyzer with cleaning, KPIs, Pareto ABC, forecasting, and charts.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/sample_sales.csv"),
        help="Path to input sales CSV (default: data/sample_sales.csv; auto-generated if missing).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write CSV summaries, Markdown report, and charts (default: output).",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Regenerate the realistic sample CSV at --input before running analysis.",
    )
    parser.add_argument("--region", type=str, default=None, help="Optional filter by region (e.g., North, East).")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional filter by product category (e.g., Electronics, Furniture).",
    )
    parser.add_argument(
        "--forecast-months",
        type=int,
        default=3,
        help="Number of future months to forecast using linear regression (default: 3).",
    )
    parser.add_argument(
        "--anomaly-z",
        type=float,
        default=2.5,
        help="Z-score threshold for flagging revenue anomalies (default: 2.5).",
    )
    parser.add_argument("--what-if-price", type=float, default=None, help="Custom what-if price change %% (e.g., 5).")
    parser.add_argument("--what-if-cost", type=float, default=None, help="Custom what-if unit cost change %% (e.g., 10).")
    parser.add_argument(
        "--what-if-discount",
        type=float,
        default=None,
        help="Custom what-if discount delta in percentage points (e.g., -3).",
    )
    parser.add_argument("--no-charts", action="store_true", help="Skip generating Matplotlib PNG charts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    if args.generate_sample or not input_path.exists():
        print(f"Generating realistic sample dataset at: {input_path}")
        generate_sample_sales_csv(input_path)

    dataset = load_sales_csv(
        input_path,
        region_filter=args.region,
        category_filter=args.category,
    )

    custom_scenario = None
    if any(v is not None for v in (args.what_if_price, args.what_if_cost, args.what_if_discount)):
        custom_scenario = (
            args.what_if_price or 0.0,
            args.what_if_cost or 0.0,
            args.what_if_discount or 0.0,
        )

    result = analyze_sales(
        dataset,
        forecast_months=args.forecast_months,
        anomaly_z_threshold=args.anomaly_z,
        custom_scenario=custom_scenario,
    )

    print(format_console_summary(dataset, result))

    output_dir: Path = args.output_dir
    csv_files = export_csv_summaries(result, output_dir)
    chart_files: list[Path] = []
    if not args.no_charts:
        chart_files = generate_charts(result, output_dir / "charts")

    report_path = export_markdown_report(
        dataset,
        result,
        output_dir / "sales_report.md",
        chart_paths=chart_files,
    )

    print("\nArtifacts exported:")
    print(f"  - Executive Report : {report_path}")
    for cf in csv_files:
        print(f"  - CSV Summary      : {cf}")
    for ch in chart_files:
        print(f"  - Chart            : {ch}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
