"""Aggregate per-variant ablation CSVs into one row-per-(variant, dataset) +
geometric-mean summary.

Run via run_ablation.sh; can also be invoked directly:
    python scripts/aggregate_ablation.py \
        --results-dir results/ablation \
        --pattern 'zero-shot-ablation-*-seed0.csv' \
        --out results/ablation/aggregate-seed0.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--pattern", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    files = sorted(args.results_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files match {args.pattern!r} in {args.results_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)

    # Geometric mean across datasets per (head_variant, model). Use
    # log-mean-exp to be robust to a single dataset accidentally being
    # zero-ish.
    def gmean(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[x > 0]
        return float(np.exp(np.log(x).mean())) if len(x) else float("nan")

    summary = (
        long.groupby(["head_variant"])
        .agg(
            n_datasets=("dataset", "count"),
            geomean_MASE=("MASE", gmean),
            geomean_WQL=("WQL", gmean),
            geomean_CRPS=("CRPS", gmean),
        )
        .reset_index()
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out.with_suffix(".per_dataset.csv"), index=False)
    summary.to_csv(args.out, index=False)

    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.per_dataset.csv')}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
