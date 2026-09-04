"""Prepare a separate BFI-2 facet dataset for the Model 2 upper-bound diagnostic.

This does not replace the deployable five-domain Model 2 dataset. It retains
the existing deterministic row splits and substitutes the 15 supplied BFI-2
facets for the five domain-average inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from src.data.prepare_bfi2_bessi import BFI2_FACET_COLUMNS, SPLITS, prepare
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from prepare_bfi2_bessi import BFI2_FACET_COLUMNS, SPLITS, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/bfi2-bessi/College student sample.xlsx"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim/bfi2-bessi-facets"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/model2_facets"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data-quality/bfi2-bessi-facets-data-quality-report.md")
    )
    parser.add_argument("--summary", type=Path, default=Path("data/metadata/bfi2-bessi-facets-preparation-summary.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare(
        input_path=args.input,
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        report_path=args.report,
        summary_path=args.summary,
        seed=args.seed,
        feature_mapping=BFI2_FACET_COLUMNS,
        feature_level="facet",
    )
    print(
        f"Prepared {summary['row_counts']['eligible_rows']} Model 2 facet rows across "
        f"{', '.join(f'{split}={summary['splits'][split]['row_count']}' for split in SPLITS)}."
    )


if __name__ == "__main__":
    main()
