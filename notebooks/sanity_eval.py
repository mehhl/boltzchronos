"""Tiny CPU sanity eval: stock amazon/chronos-t5-tiny on monash_tourism_yearly.

Lifted from scripts/evaluation/evaluate.py and trimmed to a single dataset.
Used to confirm the eval harness still runs on current dependency versions
before anyone spends GPU time. Writes the result to
scripts/evaluation/results/sanity-cpu-baseline.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from chronos import ChronosPipeline  # noqa: E402

OFFSET_TO_PERIOD = {
    "MS": "M", "ME": "M", "M": "M",
    "QS": "Q", "QE": "Q", "Q": "Q",
    "YS": "Y", "YE": "Y", "Y": "Y",
    "W": "W", "D": "D", "h": "h", "min": "min", "s": "s",
}

DATASET = dict(
    name="monash_tourism_yearly",
    hf_repo="autogluon/chronos_datasets",
    offset=-4,
    prediction_length=4,
    num_rolls=1,
)


def hf_to_gluonts(hf_dataset):
    ts_fields = [
        c for c in hf_dataset.features
        if isinstance(hf_dataset.features[c], hfds.Sequence) and c != "timestamp"
    ]
    freq = pd.infer_freq(hf_dataset[0]["timestamp"])
    freq = OFFSET_TO_PERIOD.get(freq, freq)
    out = []
    for entry in hf_dataset:
        for f in ts_fields:
            out.append({
                "start": pd.Period(entry["timestamp"][0], freq=freq),
                "target": entry[f],
            })
    return out


def make_test_data(name, hf_repo, offset, prediction_length, num_rolls=1):
    trust = hf_repo == "autogluon/chronos_datasets_extra"
    ds = hfds.load_dataset(hf_repo, name, split="train", trust_remote_code=trust)
    ds.set_format("numpy")
    gts = hf_to_gluonts(ds)
    _, tmpl = split(gts, offset=offset)
    return tmpl.generate_instances(prediction_length, windows=num_rolls)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"device={device} dtype={dtype}")

    print("loading dataset...")
    test_data = make_test_data(**DATASET)
    inputs = list(test_data.input)
    print(f"  {len(inputs)} series, prediction_length={DATASET['prediction_length']}")

    print("loading pipeline (downloads ~30MB on first run)...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map=device,
        torch_dtype=dtype,
    )

    print("predicting...")
    t0 = time.time()
    samples = []
    test_data = make_test_data(**DATASET)
    for batch in tqdm(batcher(test_data.input, batch_size=8)):
        ctx = [torch.tensor(e["target"]) for e in batch]
        out = pipeline.predict(
            ctx,
            prediction_length=DATASET["prediction_length"],
            num_samples=20,
        ).numpy()
        samples.append(out)
    samples = np.concatenate(samples)
    elapsed = time.time() - t0
    print(f"forecast time: {elapsed:.1f}s ({elapsed / max(1, len(samples)):.2f}s/series)")

    test_data = make_test_data(**DATASET)
    forecasts = [
        SampleForecast(samples=s, start_date=ts["start"] + len(ts["target"]))
        for s, ts in zip(samples, test_data.input)
    ]

    test_data = make_test_data(**DATASET)
    df = evaluate_forecasts(
        forecasts,
        test_data=test_data,
        metrics=[MASE(), MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1))],
        batch_size=5000,
    ).reset_index(drop=True)
    row = df.to_dict(orient="records")[0]
    mase = row["MASE[0.5]"]
    wql = row["mean_weighted_sum_quantile_loss"]
    print(f"\n=== stock chronos-t5-tiny on {DATASET['name']} ===")
    print(f"MASE = {mase:.4f}")
    print(f"WQL  = {wql:.4f}")
    print("(seasonal-naive baseline from results CSV: MASE=3.55, WQL=0.21)")

    out_path = REPO_ROOT / "scripts/evaluation/results/sanity-cpu-baseline.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "MASE", "WQL", "device", "elapsed_s"])
        w.writerow([
            DATASET["name"],
            "amazon/chronos-t5-tiny",
            f"{mase:.6f}",
            f"{wql:.6f}",
            device,
            f"{elapsed:.1f}",
        ])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
