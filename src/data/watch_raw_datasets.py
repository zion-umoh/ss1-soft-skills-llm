"""Watch raw tabular files and refresh their audit inventory after changes.

This is intentionally a project-local process: it runs only while the command
is active and does not install a system-wide background service.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def snapshot(data_root: Path) -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".xlsx"}
    }


def run_audit(data_root: Path, output: Path) -> None:
    audit_script = Path(__file__).with_name("audit_raw_datasets.py")
    subprocess.run(
        [
            sys.executable,
            str(audit_script),
            "--data-root",
            str(data_root),
            "--output",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/metadata/raw-data-inventory.json")
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise SystemExit(f"Raw-data directory not found: {args.data_root}")

    print("Running initial raw-data audit.", flush=True)
    run_audit(args.data_root, args.output)
    known_files = snapshot(args.data_root)
    changed_at: float | None = None
    print(f"Watching {args.data_root}; press Ctrl+C to stop.", flush=True)

    try:
        while True:
            time.sleep(args.poll_seconds)
            current_files = snapshot(args.data_root)
            if current_files != known_files:
                known_files = current_files
                changed_at = time.monotonic()
                print("Raw tabular-data change detected; waiting for files to settle.", flush=True)
            elif changed_at is not None and time.monotonic() - changed_at >= args.settle_seconds:
                print("Files settled; refreshing raw-data inventory.", flush=True)
                run_audit(args.data_root, args.output)
                changed_at = None
    except KeyboardInterrupt:
        print("Raw-data watcher stopped.", flush=True)


if __name__ == "__main__":
    main()
