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


# --------------------------------------------------------------------------
# 4. Inference-path bug + patched T5ForMeanScale
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 4. The inference-path bug

    `T5ForMeanScale.forward()` (in `scripts/training/train.py`) writes the
    censored-Gaussian *probabilities* into `outputs["logits"]`:

    ```python
    probs = self.cg_vectorized(mean, scale)         # already normalised
    probs = torch.clamp(probs, 1e-16, 1 - 1e-16)
    ...
    outputs["logits"] = probs
    return outputs
    ```

    Inference goes through `ChronosPipeline.predict()` -> `ChronosModel.forward()`
    -> `model.generate(do_sample=True, top_k=50, top_p=1.0, temperature=1.0)`.
    HuggingFace's sampling path then does, roughly:

    ```python
    next_token_logits = outputs.logits[:, -1, :]
    next_token_scores = TemperatureLogitsWarper(T)(next_token_logits)  # /= T
    next_token_scores = TopKLogitsWarper(k)(next_token_scores)
    probs = F.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, 1)
    ```

    With our `outputs.logits` already being probabilities in `[1e-16, 1]`, the
    final `softmax(probs)` gives `exp(probs) / sum(exp(probs))`. Since
    `exp(p)` for `p in [0, 1]` is in `[1, e]`, the resulting distribution is
    approximately uniform over whatever survives top-k. We end up sampling
    nearly uniformly from 50 tokens, which is a great way to torch the
    forecast.

    The fix is one line: write **log-probabilities** into `outputs["logits"]`
    instead. Then `softmax(log_probs / T)` recovers the temperature-warped
    Boltzmann distribution we actually trained.

    While we're in there:

    - hardcoded `cuda` -> use the device of the input,
    - `init_probs = tensor([0, 0, 0])` is the right *length* (the bin math
      requires three special-token slots for the default 4096 vocab) but
      using -1e9 in log space is cleaner than 1e-16 after a clamp,
    - the weights and biases of the MLP head get a slightly more careful
      init (Xavier on weights, zeros on biases).

    The architectural choice of reading `(mu, sigma)` from the 4096-d
    `lm_head` logits rather than from the 256-d decoder hidden states is
    preserved so the patched class is a drop-in replacement; switching to
    a hidden-state head is its own ablation.
"""))

CELLS.append(code(r"""
    import torch
    import torch.nn as nn
    from transformers import T5ForConditionalGeneration


    class T5ForMeanScalePatched(T5ForConditionalGeneration):
        '''Drop-in replacement for boltzchronos's T5ForMeanScale that returns
        log-probabilities so HuggingFace's sampling path works correctly.'''

        def __init__(self, config, boundaries=None, n_special_tokens=2):
            super().__init__(config)
            d_vocab = config.vocab_size

            # MLP from the categorical logits to (mu, sigma).
            self.mean_scale_head = nn.Sequential(
                nn.Linear(d_vocab, 128), nn.ReLU(),
                nn.Linear(128, 16),     nn.ReLU(),
                nn.Linear(16, 2),
            )
            for layer in (0, 2, 4):
                nn.init.xavier_uniform_(self.mean_scale_head[layer].weight)
                nn.init.zeros_(self.mean_scale_head[layer].bias)

            if boundaries is None:
                # Placeholder; train.py overwrites with the tokenizer's real
                # boundaries (length d_vocab - n_special_tokens - 1 + 2 = 4094
                # for the default 4096-token config).
                n_bin_edges = d_vocab - n_special_tokens - 1 + 2
                boundaries = torch.zeros(n_bin_edges)
            self.register_buffer("boundaries", boundaries)

            # Number of init slots needed so the final distribution covers all
            # d_vocab token IDs. With the default Chronos tokenizer this is 3
            # (pad, eos, and an unused "bucket below -1e20" slot).
            n_init = d_vocab - (boundaries.numel() - 1)
            self.register_buffer(
                "init_log_probs",
                torch.full((n_init,), -1e9),
            )

        def _censored_gaussian_logprobs(self, mu, sigma):
            '''mu, sigma: (B,) -> log-probs of shape (B, vocab).'''
            b = self.boundaries.to(mu.device).unsqueeze(0)
            sqrt2 = torch.tensor(2.0, device=mu.device).sqrt()
            cdf = 0.5 * (1 + torch.erf((b - mu.unsqueeze(1)) / (sigma.unsqueeze(1) * sqrt2)))
            bin_probs = cdf[:, 1:] - cdf[:, :-1]
            init = self.init_log_probs.to(mu.device).exp().expand(bin_probs.size(0), -1)
            probs = torch.cat([init, bin_probs], dim=1)
            probs = torch.clamp(probs, min=1e-16, max=1.0)
            return torch.log(probs)

        def forward(self, **kwargs):
            outputs = super().forward(**kwargs)
            t5_logits = outputs.logits  # (B, T, V)
            flat = t5_logits.reshape(-1, t5_logits.size(-1))
            ms = self.mean_scale_head(flat)
            mu = ms[:, 0]
            sigma = torch.relu(ms[:, 1] - 1e-10) + 1e-10
            log_probs = self._censored_gaussian_logprobs(mu, sigma)  # (B*T, V)

            labels = kwargs.get("labels", None)
            loss = None
            if labels is not None:
                loss = nn.NLLLoss(ignore_index=-100, reduction="mean")(
                    log_probs, labels.reshape(-1),
                )

            log_probs = log_probs.reshape(t5_logits.size(0), t5_logits.size(1), -1)
            outputs["loss"]   = loss
            outputs["logits"] = log_probs
            return outputs
"""))

CELLS.append(md(r"""
    Quick smoke test: load `amazon/chronos-t5-tiny` weights into the patched
    class (the MLP head will be Xavier-init since it's not in the checkpoint),
    plug it into a `ChronosPipeline`, and confirm `predict` returns finite
    forecasts of the right shape. The numbers will be junk because the head is
    random; this just verifies the inference plumbing is wired correctly.
"""))

CELLS.append(code(r"""
    from transformers import AutoConfig
    from chronos import ChronosConfig
    from chronos.chronos import ChronosModel, ChronosPipeline as ChronosPipelineCls

    def build_patched_pipeline(checkpoint="amazon/chronos-t5-tiny", device=DEVICE, dtype=DTYPE):
        cfg = AutoConfig.from_pretrained(checkpoint)
        chronos_cfg = ChronosConfig(**cfg.chronos_config)
        tokenizer = chronos_cfg.create_tokenizer()

        model = T5ForMeanScalePatched.from_pretrained(
            checkpoint, torch_dtype=dtype,
        )
        # Replace the (zero) placeholder boundaries with the real ones from the
        # tokenizer so the censored-Gaussian integrates over the right bins.
        model.boundaries = tokenizer.boundaries.to(dtype=dtype)
        model.config.chronos_config = chronos_cfg.__dict__
        model.to(device)

        return ChronosPipelineCls(
            tokenizer=tokenizer,
            model=ChronosModel(config=chronos_cfg, model=model),
        )

    smoke_pipe = build_patched_pipeline()
    ctx = torch.linspace(0, 6.28, 96).sin() * 10 + 50
    smoke_forecast = smoke_pipe.predict(
        ctx, prediction_length=4, num_samples=8,
    )
    print("forecast shape :", tuple(smoke_forecast.shape))
    print("contains NaN?  :", torch.isnan(smoke_forecast).any().item())
    print("median        :", smoke_forecast.median(dim=1).values.flatten().tolist())
    del smoke_pipe, smoke_forecast
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
