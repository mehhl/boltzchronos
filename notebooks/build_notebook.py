"""Build boltzchronos_eval.ipynb from Python source cells.

Editing notebooks as JSON by hand is painful. This script keeps each cell as a
plain Python (or Markdown) string and assembles the notebook deterministically.
Re-run after editing any of the CELLS entries.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
OUT = HERE / "boltzchronos_eval.ipynb"


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(src).strip("\n").splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(src).strip("\n").splitlines(keepends=True),
    }


CELLS: list[dict] = []


# --------------------------------------------------------------------------
# 0. Title
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    # boltzchronos: does the censored-Gaussian head help?

    This notebook walks through the boltzchronos modification of Chronos, fixes a
    couple of bugs in the original training/inference glue, and runs a small
    apples-to-apples comparison vs. stock `amazon/chronos-t5-tiny` on a single
    zero-shot dataset. It is intentionally small enough to fit on a free Colab
    T4 (16 GB VRAM) and finish in well under an hour.

    ## What's actually different about boltzchronos

    Stock Chronos: T5 decoder produces 4096 logits over a categorical bin
    vocabulary, trained with cross-entropy.

    boltzchronos (`T5ForMeanScale` in `scripts/training/train.py`): the same T5
    logits are fed into an MLP that outputs `(mu, sigma)` of a Gaussian, which is
    then *integrated between the bin boundaries* to produce a discretised
    probability distribution over the same 4096 tokens. Loss is NLL on those
    bin probabilities. The output is effectively a Boltzmann distribution
    `p(x) propto exp(-(x - mu)^2 / (2 sigma^2))` discretised to the existing token grid.

    Why might that help? Bin tokens are *ordinal* (token k corresponds to a
    real-valued bin), but the categorical softmax ignores that. A Gaussian-over-bins
    head respects ordinality and smoothness, which should help calibration and
    sample quality, especially when training data per bin is sparse.

    ## Outline

    1. **Setup** &mdash; clone repo, install deps, check the GPU.
    2. **Baseline** &mdash; eval stock `chronos-t5-tiny` on `monash_tourism_yearly`.
    3. **Bug audit** &mdash; the `T5ForMeanScale.forward` returns *probabilities* as
       `outputs.logits`, but `pipeline.predict` calls `model.generate(do_sample=True)`,
       which softmaxes them again. Inference draws from the wrong distribution.
       Patched class returns log-probabilities, makes the head device-agnostic, and
       cleans up a couple of other small things.
    4. **Train** &mdash; short (a few thousand steps) fine-tune of `chronos-t5-tiny`
       on KernelSynth using the patched head. This is *not* a publication-grade
       training run; it just tells us whether the inference path actually works
       end-to-end and gives a rough quality signal.
    5. **Compare** &mdash; eval the fine-tuned model on the same dataset and read off
       MASE / WQL / CRPS vs. the baseline.

    Skip cells you don't need. Each section is self-contained after the setup.
"""))

# --------------------------------------------------------------------------
# 1. Setup
# --------------------------------------------------------------------------
CELLS.append(md("## 1. Setup"))

CELLS.append(md(r"""
    Clone the repo on the branch with the patched code. If you've already cloned
    it, the `||` no-ops. Skip on local runs.
"""))

CELLS.append(code(r"""
    # Run only on Colab/Kaggle. Skip if you've cloned locally.
    import os
    if not os.path.isdir("boltzchronos"):
        !git clone -b claude/review-boltzmann-chronos-XB7Cj https://github.com/mehhl/boltzchronos.git
    %cd boltzchronos
"""))

CELLS.append(md(r"""
    Install the runtime deps. We pin `transformers` and `accelerate` to a known
    pair; `gluonts[pro]` is needed for the eval metrics.
"""))

CELLS.append(code(r"""
    !pip install -q \
        "torch>=2.2,<3" \
        "transformers>=4.46,<5" \
        "accelerate>=0.34" \
        "gluonts[pro]" \
        "datasets>=2.18" \
        "typer" "typer-config" "pyarrow" \
        "scipy" "scikit-learn" "joblib" "tensorboardX"

    import sys
    if "src" not in sys.path:
        sys.path.insert(0, "src")
"""))

CELLS.append(md("## 2. Environment check"))

CELLS.append(code(r"""
    import torch, sys, platform
    print("python    :", sys.version.split()[0])
    print("torch     :", torch.__version__)
    print("cuda avail:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device    :", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info(0)
        print(f"vram      : {free / 1e9:.1f} / {total / 1e9:.1f} GB free")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print("using     :", DEVICE, DTYPE)
"""))


# --------------------------------------------------------------------------
# 3. Baseline: stock chronos-t5-tiny on monash_tourism_yearly
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 3. Baseline: stock `chronos-t5-tiny` on `monash_tourism_yearly`

    `monash_tourism_yearly` is a tiny dataset (~500 series, prediction length 4)
    drawn from the published Chronos zero-shot benchmark. Cheap enough to run on
    CPU in a few minutes, useful as a sanity check that:

    - the eval harness still works on current `transformers` / `gluonts`,
    - the baseline number is recognisable (in the original paper, bigger
      Chronos models clearly beat seasonal-naive on this task).

    The number we get here is what the boltzchronos variant has to beat.
"""))

CELLS.append(md(r"""
    Eval helpers, lifted and simplified from `scripts/evaluation/evaluate.py`.
    Same metrics (MASE, WQL), same forecast-construction logic.
"""))

CELLS.append(code(r"""
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

    # Subset of pandas freq-alias mapping used by Chronos eval. Trimmed to keep
    # the cell short; extend if you swap in a dataset with a weirder freq.
    OFFSET_TO_PERIOD = {
        "MS": "M", "ME": "M", "M": "M",
        "QS": "Q", "QE": "Q", "Q": "Q",
        "YS": "Y", "YE": "Y", "Y": "Y",
        "W": "W", "D": "D", "h": "h", "min": "min", "s": "s",
    }


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


    def run_eval(pipeline, test_data, prediction_length, num_samples=20, batch_size=8):
        samples = []
        for batch in tqdm(batcher(test_data.input, batch_size=batch_size)):
            ctx = [torch.tensor(e["target"]) for e in batch]
            out = pipeline.predict(
                ctx, prediction_length=prediction_length, num_samples=num_samples,
            ).numpy()
            samples.append(out)
        samples = np.concatenate(samples)
        forecasts = [
            SampleForecast(samples=s, start_date=ts["start"] + len(ts["target"]))
            for s, ts in zip(samples, test_data.input)
        ]
        df = evaluate_forecasts(
            forecasts, test_data=test_data,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1))],
            batch_size=5000,
        ).reset_index(drop=True)
        row = df.to_dict(orient="records")[0]
        return {"MASE": row["MASE[0.5]"], "WQL": row["mean_weighted_sum_quantile_loss"]}
"""))

CELLS.append(md(r"""
    Pick the dataset, build the test split, and run stock `chronos-t5-tiny` on it.
    The seasonal-naive baseline numbers from the original eval are included for
    quick comparison: `MASE = 3.55`, `WQL = 0.21`.
"""))

CELLS.append(code(r"""
    from chronos import ChronosPipeline

    DATASET = dict(
        name="monash_tourism_yearly",
        hf_repo="autogluon/chronos_datasets",
        offset=-4,
        prediction_length=4,
        num_rolls=1,
    )
    test_data = make_test_data(**DATASET)
    n_series = len(list(test_data.input))
    print(f"{DATASET['name']}: {n_series} series, prediction_length={DATASET['prediction_length']}")

    baseline_pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny", device_map=DEVICE, torch_dtype=DTYPE,
    )

    # Rebuild test_data because generate_instances() was consumed above when we
    # counted the series length.
    test_data = make_test_data(**DATASET)
    baseline_metrics = run_eval(
        baseline_pipe, test_data, DATASET["prediction_length"],
        num_samples=20, batch_size=16,
    )
    print("stock chronos-t5-tiny:", baseline_metrics)
    print("seasonal-naive (from results CSV): MASE=3.55, WQL=0.21")
"""))


def write_notebook():
    for i, cell in enumerate(CELLS):
        cell.setdefault("id", f"cell-{i:03d}")
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} ({len(CELLS)} cells, {OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    write_notebook()
