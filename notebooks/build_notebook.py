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
    <a href="https://colab.research.google.com/github/mehhl/boltzchronos/blob/claude/complete-task-md-v8TeV/notebooks/boltzchronos_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
        !git clone -b claude/complete-task-md-v8TeV https://github.com/mehhl/boltzchronos.git
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


    def to_period_freq(freq):
        # pd.infer_freq returns 'YE-DEC' but pd.Period wants 'Y-DEC'.
        if freq is None or freq in OFFSET_TO_PERIOD:
            return OFFSET_TO_PERIOD.get(freq, freq)
        base, _, anchor = freq.partition("-")
        if base in OFFSET_TO_PERIOD:
            period_base = OFFSET_TO_PERIOD[base]
            return f"{period_base}-{anchor}" if anchor else period_base
        return freq


    def hf_to_gluonts(hf_dataset):
        ts_fields = [
            c for c in hf_dataset.features
            if isinstance(hf_dataset.features[c], hfds.Sequence) and c != "timestamp"
        ]
        freq = pd.infer_freq(hf_dataset[0]["timestamp"])
        freq = to_period_freq(freq)
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


# --------------------------------------------------------------------------
# 5. Generate a small KernelSynth dataset
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 5. Tiny KernelSynth dataset

    The original Chronos training corpus is ~85B tokens of TSMixup + KernelSynth.
    For a sniff test we just want a few thousand synthetic GP-prior series so the
    fine-tune actually has something to learn from. `scripts/kernel-synth.py`
    is the upstream generator; we re-use the same kernel bank but dial it down to
    `N_SYNTH` series of length 1024.

    On a Colab T4 with 4 CPU cores, generating ~3000 series takes a few minutes.
"""))

CELLS.append(code(r"""
    import sys
    sys.path.insert(0, "scripts")
    from importlib import reload

    import kernel_synth_mod  # placeholder so the next import works after reload
    """ + "" + r"""
"""))
# That import dance is silly; let's do a cleaner version that imports the
# generator directly from the script as a module.

# Replace the placeholder with a real cell.
CELLS.pop()
CELLS.append(code(r"""
    # Reuse the upstream generator from scripts/kernel-synth.py.
    # The script's filename has a hyphen so we load it manually.
    import importlib.util, sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "kernel_synth", Path("scripts/kernel-synth.py").resolve()
    )
    kernel_synth = importlib.util.module_from_spec(spec)
    sys.modules["kernel_synth"] = kernel_synth
    spec.loader.exec_module(kernel_synth)

    print("KERNEL_BANK has", len(kernel_synth.KERNEL_BANK), "kernels; LENGTH =", kernel_synth.LENGTH)
"""))

CELLS.append(md(r"""
    Generate the synthetic series in parallel and dump them to a GluonTS arrow
    file under `data/kernelsynth-tiny.arrow`. Tweak `N_SYNTH` if you want to
    spend more or less compute. With `N_SYNTH = 3000` and `MAX_KERNELS = 4` the
    file is ~30 MB.
"""))

CELLS.append(code(r"""
    import os
    from pathlib import Path
    import numpy as np
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from gluonts.dataset.arrow import ArrowWriter

    N_SYNTH      = 3000   # ~3000 series ~= ~3M scalar steps
    MAX_KERNELS  = 4
    DATA_DIR     = Path("data"); DATA_DIR.mkdir(exist_ok=True)
    SYNTH_PATH   = DATA_DIR / "kernelsynth-tiny.arrow"

    if SYNTH_PATH.exists():
        print(f"{SYNTH_PATH} already exists ({SYNTH_PATH.stat().st_size / 1e6:.1f} MB) - skipping generation")
    else:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
        series = Parallel(n_jobs=n_jobs)(
            delayed(kernel_synth.generate_time_series)(max_kernels=MAX_KERNELS)
            for _ in tqdm(range(N_SYNTH))
        )
        ArrowWriter(compression="lz4").write_to_file(series, path=SYNTH_PATH)
        print(f"wrote {SYNTH_PATH} ({SYNTH_PATH.stat().st_size / 1e6:.1f} MB)")
"""))


# --------------------------------------------------------------------------
# 6. Short fine-tune
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 6. Short fine-tune of the patched head

    We start from `amazon/chronos-t5-tiny`, swap in `T5ForMeanScalePatched`, and
    fine-tune for `MAX_STEPS` steps on the synthetic dataset above. The MLP head
    is randomly initialised (Xavier), the rest of the T5 weights are pre-trained.

    Calibrate `MAX_STEPS` to your compute budget:

    - 500 steps   - few minutes on a T4, basically just smoke test
    - 2000 steps  - ~20 min on a T4, you can already see whether loss is going down
    - 5000 steps  - ~50 min on a T4, the regime where eval numbers might mean something

    The point of this run is not to beat Chronos. It's to verify the inference
    path works end-to-end and produce a checkpoint we can compare against the
    stock baseline on the same dataset.
"""))

CELLS.append(code(r"""
    from functools import partial

    import numpy as np
    from gluonts.dataset.common import FileDataset
    from gluonts.itertools import Cyclic, Map, Filter
    from gluonts.transform import (
        ExpectedNumInstanceSampler,
        FilterTransformation,
        InstanceSplitter,
        LeavesMissingValues,
    )
    from torch.utils.data import IterableDataset, get_worker_info
    from chronos import ChronosTokenizer


    # Trimmed-down version of train.py's ChronosDataset, just enough for the
    # short fine-tune. No multi-dataset mixing, no shuffle buffer, no causal
    # path; we only need the seq2seq + KernelSynth case.
    class TinyChronosDataset(IterableDataset):
        def __init__(self, file_path, tokenizer, context_length=512,
                     prediction_length=64, min_past=64, drop_prob=0.2):
            self.dataset = Filter(
                lambda e: len(e["target"]) >= min_past + prediction_length,
                FileDataset(path=Path(file_path), freq="h"),
            )
            self.tokenizer = tokenizer
            self.context_length = context_length
            self.prediction_length = prediction_length
            self.min_past = min_past
            self.drop_prob = drop_prob

        def _splitter(self):
            return InstanceSplitter(
                target_field="target",
                is_pad_field="is_pad",
                start_field="start",
                forecast_start_field="forecast_start",
                instance_sampler=ExpectedNumInstanceSampler(
                    num_instances=1.0, min_instances=1,
                    min_past=self.min_past, min_future=self.prediction_length,
                ),
                past_length=self.context_length,
                future_length=self.prediction_length,
                dummy_value=np.nan,
            ) + FilterTransformation(
                condition=lambda e: (~np.isnan(e["past_target"])).sum() > 0
            )

        def _to_hf(self, entry):
            past   = torch.tensor(entry["past_target"]).unsqueeze(0)
            future = torch.tensor(entry["future_target"]).unsqueeze(0)
            input_ids, attn_mask, scale = self.tokenizer.context_input_transform(past)
            labels, labels_mask          = self.tokenizer.label_input_transform(future, scale)
            labels[labels_mask == 0] = -100
            return {
                "input_ids":      input_ids.squeeze(0),
                "attention_mask": attn_mask.squeeze(0),
                "labels":         labels.squeeze(0),
            }

        def __iter__(self):
            data = Cyclic(self.dataset)
            data = Map(self._preprocess, data)
            data = self._splitter().apply(data, is_train=True)
            for entry in data:
                yield self._to_hf(entry)

        def _preprocess(self, entry):
            target = np.asarray(entry["target"], dtype=np.float32)
            if self.drop_prob > 0:
                p = np.random.uniform(0.0, self.drop_prob)
                mask = np.random.choice([True, False], size=len(target), p=[p, 1 - p])
                target = target.copy()
                target[mask] = np.nan
            return {"start": entry["start"], "target": target}
"""))

CELLS.append(code(r"""
    from transformers import Trainer, TrainingArguments

    MAX_STEPS  = 2000
    BATCH_SIZE = 16 if torch.cuda.is_available() else 2
    CTX_LEN    = 512
    PRED_LEN   = 64
    LR         = 5e-4

    chronos_cfg_dict = AutoConfig.from_pretrained("amazon/chronos-t5-tiny").chronos_config
    chronos_cfg      = ChronosConfig(**chronos_cfg_dict)
    train_tokenizer  = chronos_cfg.create_tokenizer()

    train_ds = TinyChronosDataset(
        file_path=str(SYNTH_PATH),
        tokenizer=train_tokenizer,
        context_length=CTX_LEN,
        prediction_length=PRED_LEN,
        min_past=PRED_LEN,
        drop_prob=0.2,
    )

    # Patched model from the stock checkpoint, with real boundaries plugged in.
    train_model = T5ForMeanScalePatched.from_pretrained(
        "amazon/chronos-t5-tiny", torch_dtype=torch.float32,
    )
    train_model.boundaries = train_tokenizer.boundaries.float()
    train_model.config.chronos_config = chronos_cfg.__dict__


    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **_):
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss


    args = TrainingArguments(
        output_dir="output/run-patched",
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        bf16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        # torch_compile=True can help on a fresh GPU, off by default to avoid
        # warmup compile time eating the wallclock budget.
        torch_compile=False,
    )

    trainer = CustomTrainer(model=train_model, args=args, train_dataset=train_ds)
    trainer.train()

    # Persist the run so we can rebuild a pipeline from it.
    train_model.save_pretrained("output/run-patched/checkpoint-final")
    print("saved -> output/run-patched/checkpoint-final")
"""))


# --------------------------------------------------------------------------
# 7. Eval the patched fine-tune + comparison
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 7. Eval the patched model on the same dataset

    Same dataset, same eval helper as the baseline. Build a fresh
    `ChronosPipeline` around the fine-tuned checkpoint and read off MASE/WQL.
    Also compute CRPS by hand from the sample forecasts &mdash; it's the metric
    where a parametric calibrated head is most likely to win, and the original
    eval script doesn't report it.
"""))

CELLS.append(code(r"""
    def crps_from_samples(samples, target):
        '''Approximate continuous ranked probability score from MC samples.

        samples: (n_series, n_samples, horizon)
        target : (n_series, horizon)
        Returns the average CRPS over series and horizon.
        '''
        s = np.asarray(samples, dtype=np.float64)
        y = np.asarray(target,  dtype=np.float64)[:, None, :]
        n = s.shape[1]
        # E|X - y|
        term1 = np.mean(np.abs(s - y), axis=1)
        # 0.5 * E|X - X'| via sorted-sample formula
        s_sorted = np.sort(s, axis=1)
        weights = (2 * np.arange(1, n + 1) - n - 1)
        term2 = (weights[None, :, None] * s_sorted).sum(axis=1) / (n * n)
        return float(np.mean(term1 - term2))


    def run_full_eval(pipeline, dataset_cfg, num_samples=20, batch_size=16):
        td = make_test_data(**dataset_cfg)
        pred_len = dataset_cfg["prediction_length"]
        all_samples, all_targets = [], []
        for batch in tqdm(batcher(td.input, batch_size=batch_size)):
            ctx = [torch.tensor(e["target"]) for e in batch]
            out = pipeline.predict(ctx, prediction_length=pred_len, num_samples=num_samples).numpy()
            all_samples.append(out)
        forecast_samples = np.concatenate(all_samples)

        # Re-iterate test data (it's a generator-backed object) for forecasts + targets
        td2 = make_test_data(**dataset_cfg)
        forecasts, targets = [], []
        for s, ts, label in zip(forecast_samples, td2.input, make_test_data(**dataset_cfg).label):
            forecasts.append(SampleForecast(samples=s, start_date=ts["start"] + len(ts["target"])))
            targets.append(np.asarray(label["target"], dtype=np.float64))

        td3 = make_test_data(**dataset_cfg)
        df = evaluate_forecasts(
            forecasts, test_data=td3,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1))],
            batch_size=5000,
        ).reset_index(drop=True)
        row = df.to_dict(orient="records")[0]

        crps = crps_from_samples(forecast_samples, np.stack(targets))
        return {
            "MASE": row["MASE[0.5]"],
            "WQL":  row["mean_weighted_sum_quantile_loss"],
            "CRPS": crps,
        }
"""))

CELLS.append(code(r"""
    # Rebuild the baseline forecasts with the same eval (now also CRPS).
    baseline_full = run_full_eval(baseline_pipe, DATASET, num_samples=20, batch_size=16)
    print("baseline (stock chronos-t5-tiny):", baseline_full)

    # Patched-and-fine-tuned pipeline.
    patched_pipe = build_patched_pipeline(checkpoint="output/run-patched/checkpoint-final")
    patched_full = run_full_eval(patched_pipe, DATASET, num_samples=20, batch_size=16)
    print("patched + fine-tuned         :", patched_full)
"""))

CELLS.append(md(r"""
    ## 8. Side-by-side and what to read into it

    A short fine-tune on KernelSynth alone is *not* a fair comparison vs. the
    original Chronos model that saw ~85B tokens of TSMixup + KernelSynth. So
    don't expect a clean win. What this run *can* tell you:

    1. **Does the patched inference path produce a sane distribution?** If
       MASE is wildly worse than the baseline (say, 10x), something's still
       broken in the head. If it's the same order of magnitude, the plumbing
       works.
    2. **Is loss going down?** Watch the trainer log output. NLL on the
       censored Gaussian should drop monotonically; if it plateaus at the
       initial value, the MLP head can't learn from the lm_head logits and
       you should switch to projecting from the decoder hidden states.
    3. **Calibration delta on CRPS**, even with a tiny fine-tune. The
       parametric head's biggest theoretical advantage is calibration; if
       CRPS is competitive even when MASE isn't, that's evidence the
       direction is sound and worth a real (multi-GPU, full corpus)
       training run.

    Next experiments worth running, in roughly increasing cost:

    - **head from hidden states** &mdash; replace `Linear(d_vocab, 128)` with
      `Linear(d_model, 128)` and read from `decoder_outputs.last_hidden_state`
      instead of `outputs.logits`. Cleaner architecture, fewer parameters,
      should learn faster.
    - **mixture of K Gaussians** &mdash; closer in expressivity to the
      4096-way softmax while keeping ordinality. Output `(K, mu_K, sigma_K, weight_K)`.
    - **learnable bin boundaries** &mdash; the original code hints at this
      (`trained_boundaries = model.boundaries` at end of train.py) but the
      buffer is `requires_grad=False`. Flip that and let the boundaries move.
    - **full-corpus pretraining** of the patched architecture on a single
      A100 for ~12-24h vs. stock T5-tiny CE pretraining of the same length.
      That's the apples-to-apples experiment that would actually answer
      "is this an improvement on Chronos?".
"""))

CELLS.append(code(r"""
    import pandas as pd
    summary = pd.DataFrame([
        {"model": "stock chronos-t5-tiny", **baseline_full},
        {"model": "patched + fine-tuned",  **patched_full},
    ])
    summary
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
