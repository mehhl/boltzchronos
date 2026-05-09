"""Build boltzchronos_eval_v2.ipynb.

Same general flow as v1 (build_notebook.py), but:

- KernelSynth dialed down to 500 series (was 3000) so the run finishes
  faster -- saves ~40 min of the v1 wallclock
- Fine-tune bumped to 4000 steps (was 2000) so the head has more chance
  to learn something meaningful
- Eval is on 6 zero-shot datasets instead of 1, so we have a chance of
  finding any dataset where patched wins on anything
- Calibration diagnostics: empirical 80%-interval coverage and sharpness
  per dataset. Calibration is the parametric head's theoretical
  advantage -- if a "hint of being markedly better at SOMETHING" exists,
  it should show up here, not in MASE.
- Honest summary cell: if there's no signal, says so.

Re-run after editing any of the CELLS entries.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
OUT = HERE / "boltzchronos_eval_v2.ipynb"


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
    <a href="https://colab.research.google.com/github/mehhl/boltzchronos/blob/claude/complete-task-md-v8TeV/notebooks/boltzchronos_eval_v2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

    # boltzchronos v2: looking for a hint of advantage

    v1 of this notebook shipped a clean apples-to-apples on one dataset and
    found the patched model ~3-6% behind stock chronos-t5-tiny on every
    metric. That's the textbook outcome for a 2000-step KernelSynth fine-tune
    of a randomly-initialised MLP head bolted onto a model whose backbone
    saw 85B tokens of TSMixup+KernelSynth in pretraining -- it tells us
    nothing's broken, but it also gives us no reason to spend more compute.

    This notebook is asking a sharper question: **is there any single thing
    boltzchronos does markedly better than stock?** The point is to either
    find a hint that justifies a real (multi-day, multi-GPU) training run,
    or to find that there isn't one and stop early.

    ## Where to look

    Point accuracy (MASE) is *not* where the patched head should beat stock.
    The MLP head is the part that's barely trained, and a poorly-trained
    parametric head producing the wrong mean is going to lose to a
    well-trained categorical that maps to the right token.

    Where the parametric head has a *theoretical* advantage:

    1. **Calibration of prediction intervals.** A categorical softmax can
       happily put high probability on bins far from the true value with no
       penalty for ordinal distance. A Gaussian head can't -- mass is
       contiguous around `mu`. So 80%-interval coverage and sharpness are
       the diagnostics worth checking.
    2. **Per-quantile WQL.** WQL is averaged across q=0.1..0.9. If the
       patched head is uniformly worse on the median (0.5) but
       competitive or better on tail quantiles (0.1, 0.9), that's a
       calibration story even when the aggregate WQL is worse.
    3. **Dataset diversity.** Some datasets favor sharp categorical
       predictions (data lives at specific bin centres); others favor
       smooth distributions (continuous targets that fall between bins).
       Run a handful of datasets, see if any go the patched way.

    ## What we'll do

    1. Setup -- same as v1.
    2. Quick smoke (pinned to v1) so we know the patched class still works.
    3. Tiny KernelSynth (500 series, ~8 min on T4 CPU) and a 4000-step
       fine-tune (~14 min on T4 GPU) of T5ForMeanScalePatched.
    4. Eval BOTH stock and patched on six zero-shot datasets:
       monash_tourism_yearly, exchange_rate, monash_covid_deaths,
       monash_nn5_weekly, monash_m1_yearly, ercot. Collect raw forecast
       samples so we can compute calibration metrics.
    5. Calibration diagnostics: empirical 80%-interval coverage,
       sharpness (interval width), per-quantile WQL breakdown.
    6. Honest summary: where (if anywhere) does patched win, and is it
       enough to argue for more compute?

    Estimated wallclock on a free Colab T4: ~40 min. Roughly half v1.
"""))


# --------------------------------------------------------------------------
# 1. Setup
# --------------------------------------------------------------------------
CELLS.append(md("## 1. Setup"))

CELLS.append(code(r"""
    # Run only on Colab/Kaggle. Skip if you've cloned locally.
    import os
    if not os.path.isdir("boltzchronos"):
        !git clone -b claude/complete-task-md-v8TeV https://github.com/mehhl/boltzchronos.git
    %cd boltzchronos
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

    # Disable HF's xet protocol -- it returns 503/416 on chronos-t5-tiny
    # and leaves the download hung. Classic resolve protocol works fine.
    os.environ["HF_HUB_DISABLE_XET"] = "1"

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
# 3. Eval helpers (multi-dataset + calibration)
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 3. Eval helpers: multi-dataset + calibration

    Lifted from `scripts/evaluation/evaluate.py` and v1 of this notebook.
    Two new helpers vs v1: `interval_coverage_and_sharpness` (empirical
    P[target in interval] + mean width) and `per_quantile_loss` (WQL
    broken out per quantile rather than averaged).
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

    OFFSET_TO_PERIOD = {
        "MS": "M", "ME": "M", "M": "M",
        "QS": "Q", "QE": "Q", "Q": "Q",
        "YS": "Y", "YE": "Y", "Y": "Y",
        "W": "W", "D": "D", "h": "h", "min": "min", "s": "s",
    }


    def to_period_freq(freq):
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


    def crps_from_samples(samples, target):
        s = np.asarray(samples, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)[:, None, :]
        n = s.shape[1]
        term1 = np.mean(np.abs(s - y), axis=1)
        s_sorted = np.sort(s, axis=1)
        weights = (2 * np.arange(1, n + 1) - n - 1)
        term2 = (weights[None, :, None] * s_sorted).sum(axis=1) / (n * n)
        return float(np.mean(term1 - term2))


    def interval_coverage_and_sharpness(samples, target, lower=0.1, upper=0.9):
        '''Empirical P[target in [Q_lower, Q_upper]] and mean interval width.

        samples: (n_series, n_samples, horizon)
        target : (n_series, horizon)
        Returns (coverage, mean_width). For nominal (1 - 2*lower) coverage
        a well-calibrated model should give coverage ~= 1 - 2*lower.
        '''
        s = np.asarray(samples, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        q_lo = np.quantile(s, lower, axis=1)   # (n_series, horizon)
        q_hi = np.quantile(s, upper, axis=1)
        inside = (y >= q_lo) & (y <= q_hi)
        coverage = float(inside.mean())
        mean_width = float((q_hi - q_lo).mean())
        return coverage, mean_width


    def per_quantile_loss(samples, target, quantiles=tuple(np.arange(0.1, 1.0, 0.1))):
        '''Per-quantile pinball loss, normalised by mean(|target|) (the
        same denominator MeanWeightedSumQuantileLoss uses, so summing
        equals 9 * the WQL the eval script reports).
        '''
        s = np.asarray(samples, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)[:, None, :]
        denom = np.abs(y).mean()
        out = {}
        for q in quantiles:
            qs = np.quantile(s, q, axis=1, keepdims=True)
            loss = np.where(y >= qs, q * (y - qs), (1 - q) * (qs - y))
            out[round(float(q), 1)] = float(loss.mean() / denom)
        return out


    def run_full_eval(pipeline, dataset_cfg, num_samples=20, batch_size=16):
        '''Run forecasts + return MASE/WQL/CRPS plus the raw forecast
        samples and matching targets for downstream calibration analysis.
        '''
        td = make_test_data(**dataset_cfg)
        pred_len = dataset_cfg["prediction_length"]
        all_samples = []
        for batch in tqdm(batcher(td.input, batch_size=batch_size),
                          desc=dataset_cfg["name"], leave=False):
            ctx = [torch.tensor(e["target"]) for e in batch]
            out = pipeline.predict(
                ctx, prediction_length=pred_len, num_samples=num_samples,
            ).numpy()
            all_samples.append(out)
        forecast_samples = np.concatenate(all_samples)

        td2 = make_test_data(**dataset_cfg)
        td3 = make_test_data(**dataset_cfg)
        forecasts, targets = [], []
        for s, ts, label in zip(forecast_samples, td2.input, td3.label):
            forecasts.append(SampleForecast(samples=s, start_date=ts["start"] + len(ts["target"])))
            targets.append(np.asarray(label["target"], dtype=np.float64))
        targets = np.stack(targets)

        td4 = make_test_data(**dataset_cfg)
        df = evaluate_forecasts(
            forecasts, test_data=td4,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1))],
            batch_size=5000,
        ).reset_index(drop=True)
        row = df.to_dict(orient="records")[0]

        crps = crps_from_samples(forecast_samples, targets)
        cov80, width80 = interval_coverage_and_sharpness(forecast_samples, targets, 0.1, 0.9)
        cov50, width50 = interval_coverage_and_sharpness(forecast_samples, targets, 0.25, 0.75)
        pql = per_quantile_loss(forecast_samples, targets)

        return {
            "MASE": float(row["MASE[0.5]"]),
            "WQL":  float(row["mean_weighted_sum_quantile_loss"]),
            "CRPS": crps,
            "cov80": cov80,
            "width80": width80,
            "cov50": cov50,
            "width50": width50,
            **{f"pql_{k:.1f}": v for k, v in pql.items()},
        }
"""))


# --------------------------------------------------------------------------
# 4. Patched class + pipeline builder
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 4. Patched class

    Same `T5ForMeanScalePatched` as v1, including the meta-tensor
    `_init_weights` fix and the off-by-one boundaries fix that were
    debugged in the CPU mini-eval. See v1 (cell on "the inference-path
    bug") for the long story.
"""))

CELLS.append(code(r"""
    import torch
    import torch.nn as nn
    from transformers import T5ForConditionalGeneration


    class T5ForMeanScalePatched(T5ForConditionalGeneration):
        def __init__(self, config, boundaries=None, n_special_tokens=2):
            super().__init__(config)
            d_vocab = config.vocab_size
            self.mean_scale_head = nn.Sequential(
                nn.Linear(d_vocab, 128), nn.ReLU(),
                nn.Linear(128, 16),     nn.ReLU(),
                nn.Linear(16, 2),
            )
            for layer in self.mean_scale_head:
                if isinstance(layer, nn.Linear):
                    layer._is_mean_scale_head = True
            self._init_mean_scale_head()
            if boundaries is None:
                n_bin_edges = d_vocab - n_special_tokens
                boundaries = torch.zeros(n_bin_edges)
            self.register_buffer("boundaries", boundaries, persistent=False)
            n_init = d_vocab - (boundaries.numel() - 1)
            self.register_buffer(
                "init_log_probs", torch.full((n_init,), -1e9), persistent=False,
            )

        def _init_mean_scale_head(self):
            for layer in self.mean_scale_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        def _init_weights(self, module):
            super()._init_weights(module)
            if isinstance(module, nn.Linear) and getattr(module, "_is_mean_scale_head", False):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def _censored_gaussian_logprobs(self, mu, sigma):
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
            t5_logits = outputs.logits
            flat = t5_logits.reshape(-1, t5_logits.size(-1))
            ms = self.mean_scale_head(flat)
            mu = ms[:, 0]
            sigma = torch.relu(ms[:, 1] - 1e-10) + 1e-10
            log_probs = self._censored_gaussian_logprobs(mu, sigma)
            labels = kwargs.get("labels", None)
            loss = None
            if labels is not None:
                loss = nn.NLLLoss(ignore_index=-100, reduction="mean")(
                    log_probs, labels.reshape(-1),
                )
            log_probs = log_probs.reshape(t5_logits.size(0), t5_logits.size(1), -1)
            outputs["loss"] = loss
            outputs["logits"] = log_probs
            return outputs


    from transformers import AutoConfig
    from chronos import ChronosConfig
    from chronos.chronos import ChronosModel, ChronosPipeline as ChronosPipelineCls

    def build_patched_pipeline(checkpoint, device=DEVICE, dtype=DTYPE):
        cfg = AutoConfig.from_pretrained(checkpoint)
        chronos_cfg = ChronosConfig(**cfg.chronos_config)
        tokenizer = chronos_cfg.create_tokenizer()
        model = T5ForMeanScalePatched.from_pretrained(checkpoint, torch_dtype=dtype)
        model.boundaries = tokenizer.boundaries.to(dtype=dtype)
        model.config.chronos_config = chronos_cfg.__dict__
        model.to(device)
        return ChronosPipelineCls(
            tokenizer=tokenizer,
            model=ChronosModel(config=chronos_cfg, model=model),
        )
"""))


# --------------------------------------------------------------------------
# 5. KernelSynth (smaller than v1)
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 5. Tiny KernelSynth (500 series, ~8 min)

    v1 used 3000 series and KernelSynth gen ate 47 min of the 57 min
    wallclock. With batch=16 and 4000 steps, the trainer needs only
    ~64k samples; even 500 length-1024 series gives orders of magnitude
    more sliding-window samples than that. We don't need 3000.
"""))

CELLS.append(code(r"""
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

CELLS.append(code(r"""
    import os
    from pathlib import Path
    import numpy as np
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from gluonts.dataset.arrow import ArrowWriter

    N_SYNTH      = 500
    MAX_KERNELS  = 4
    DATA_DIR     = Path("data"); DATA_DIR.mkdir(exist_ok=True)
    SYNTH_PATH   = DATA_DIR / "kernelsynth-v2.arrow"

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
# 6. Fine-tune (4000 steps, was 2000 in v1)
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 6. Fine-tune the patched head (4000 steps, ~14 min)

    Doubled vs v1 to give the head a fairer shot. Loss in v1 was still
    decreasing past step 2000 (11.28 -> 6.03 by step 350 and continuing),
    so more steps should help -- though we're still nowhere near
    saturating.
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
    )
    from torch.utils.data import IterableDataset

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

    MAX_STEPS  = 4000
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
        output_dir="output/run-patched-v2",
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=100,
        report_to=[],
        bf16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        torch_compile=False,
    )

    trainer = CustomTrainer(model=train_model, args=args, train_dataset=train_ds)
    trainer.train()
    train_model.save_pretrained("output/run-patched-v2/checkpoint-final")
    print("saved -> output/run-patched-v2/checkpoint-final")
"""))


# --------------------------------------------------------------------------
# 7. Multi-dataset eval
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 7. Eval both arms on six zero-shot datasets

    Six datasets that vary in horizon, frequency, and series count:

    | Dataset                  | Freq   | Pred horizon | ~Series |
    | ------------------------ | ------ | ------------ | ------- |
    | monash_tourism_yearly    | yearly | 4            | 518     |
    | exchange_rate            | daily  | 30           | 8       |
    | monash_covid_deaths      | daily  | 30           | 266     |
    | monash_nn5_weekly        | weekly | 8            | 111     |
    | monash_m1_yearly         | yearly | 6            | 181     |
    | ercot                    | hourly | 24           | 8       |

    Same eval harness for both stock and patched, same num_samples, same
    batch size. Saves the raw sample arrays for downstream calibration.
"""))

CELLS.append(code(r"""
    from chronos import ChronosPipeline

    DATASETS = [
        dict(name="monash_tourism_yearly",  hf_repo="autogluon/chronos_datasets", offset=-4,  prediction_length=4,  num_rolls=1),
        dict(name="exchange_rate",          hf_repo="autogluon/chronos_datasets", offset=-30, prediction_length=30, num_rolls=1),
        dict(name="monash_covid_deaths",    hf_repo="autogluon/chronos_datasets", offset=-30, prediction_length=30, num_rolls=1),
        dict(name="monash_nn5_weekly",      hf_repo="autogluon/chronos_datasets", offset=-8,  prediction_length=8,  num_rolls=1),
        dict(name="monash_m1_yearly",       hf_repo="autogluon/chronos_datasets", offset=-6,  prediction_length=6,  num_rolls=1),
        dict(name="ercot",                  hf_repo="autogluon/chronos_datasets", offset=-24, prediction_length=24, num_rolls=1),
    ]

    print("Loading stock chronos-t5-tiny...")
    stock_pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny", device_map=DEVICE, torch_dtype=DTYPE,
    )

    print("Building patched pipeline from fine-tuned checkpoint...")
    patched_pipe = build_patched_pipeline("output/run-patched-v2/checkpoint-final")

    rows = []
    for d in DATASETS:
        print(f"\n=== {d['name']} ===")
        stock_m = run_full_eval(stock_pipe, d, num_samples=20, batch_size=16)
        patched_m = run_full_eval(patched_pipe, d, num_samples=20, batch_size=16)
        rows.append({"dataset": d["name"], "model": "stock", **stock_m})
        rows.append({"dataset": d["name"], "model": "patched", **patched_m})
        print(f"  stock  : MASE={stock_m['MASE']:.3f} WQL={stock_m['WQL']:.3f} CRPS={stock_m['CRPS']:.1f} cov80={stock_m['cov80']:.3f}")
        print(f"  patched: MASE={patched_m['MASE']:.3f} WQL={patched_m['WQL']:.3f} CRPS={patched_m['CRPS']:.1f} cov80={patched_m['cov80']:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv("results-v2.csv", index=False)
    print("\nWrote results-v2.csv")
    results
"""))


# --------------------------------------------------------------------------
# 8. Headline analysis
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 8. Where, if anywhere, does patched win?

    Three lenses:

    1. **Per-dataset point accuracy** -- does patched beat stock on
       MASE/WQL/CRPS on any dataset?
    2. **Calibration** -- nominal 80% interval should give 0.80 coverage.
       Whichever model is closer to 0.80 (in absolute terms) wins. If
       they're both close, sharper (smaller width80) wins, since both
       hit nominal coverage.
    3. **Per-quantile WQL** -- patched might be uniformly worse on the
       median (0.5) but competitive on tails (0.1, 0.9). That's a
       calibration story even when the aggregate WQL is worse.
"""))

CELLS.append(code(r"""
    # Lens 1: point accuracy
    point_accuracy = results.pivot(index="dataset", columns="model",
                                   values=["MASE", "WQL", "CRPS"])
    # Compute relative gap: (patched - stock) / stock
    for m in ("MASE", "WQL", "CRPS"):
        point_accuracy[(m, "rel_gap_pct")] = (
            (point_accuracy[(m, "patched")] - point_accuracy[(m, "stock")])
            / point_accuracy[(m, "stock")] * 100
        )
    print("Per-dataset point accuracy (negative rel_gap = patched WINS):")
    point_accuracy.round(3)
"""))

CELLS.append(code(r"""
    # Lens 2: calibration
    cal = results.pivot(index="dataset", columns="model",
                        values=["cov80", "width80", "cov50", "width50"])
    # |coverage - nominal|: smaller is better
    cal[("cov80_err", "stock")]   = (cal[("cov80", "stock")]   - 0.80).abs()
    cal[("cov80_err", "patched")] = (cal[("cov80", "patched")] - 0.80).abs()
    cal[("cov50_err", "stock")]   = (cal[("cov50", "stock")]   - 0.50).abs()
    cal[("cov50_err", "patched")] = (cal[("cov50", "patched")] - 0.50).abs()
    print("Calibration:")
    print("  cov80 = empirical P[target in [Q10, Q90]]; nominal 0.80")
    print("  cov80_err = |cov80 - 0.80|; smaller is better-calibrated")
    print("  width80 = mean width of [Q10, Q90]; smaller is sharper "
          "(only meaningful at similar coverage)")
    cal.round(3)
"""))

CELLS.append(code(r"""
    # Lens 3: per-quantile WQL breakdown
    qcols = [f"pql_{q:.1f}" for q in np.arange(0.1, 1.0, 0.1)]
    pq = results.pivot(index="dataset", columns="model", values=qcols)
    # Reorder columns so each quantile shows stock vs patched side by side
    new_cols = [(q, m) for q in qcols for m in ("stock", "patched")]
    pq = pq[new_cols]
    print("Per-quantile pinball loss (lower = better; sum over q ~= 9 * WQL):")
    pq.round(4)
"""))

CELLS.append(code(r"""
    # Headline: count wins
    wins = []
    for m in ("MASE", "WQL", "CRPS"):
        for ds in results.dataset.unique():
            sub = results[results.dataset == ds]
            stock_v = sub[sub.model == "stock"][m].iloc[0]
            patched_v = sub[sub.model == "patched"][m].iloc[0]
            if patched_v < stock_v:
                rel = (stock_v - patched_v) / stock_v * 100
                wins.append({"metric": m, "dataset": ds, "patched_better_by_pct": round(rel, 2)})

    cal_wins = []
    for ds in results.dataset.unique():
        sub = results[results.dataset == ds]
        stock_err = abs(sub[sub.model == "stock"]["cov80"].iloc[0] - 0.80)
        patched_err = abs(sub[sub.model == "patched"]["cov80"].iloc[0] - 0.80)
        if patched_err < stock_err:
            cal_wins.append({"dataset": ds,
                             "stock_cov80_err": round(stock_err, 4),
                             "patched_cov80_err": round(patched_err, 4),
                             "improvement": round(stock_err - patched_err, 4)})

    print("=== POINT-ACCURACY WINS for patched ===")
    if wins:
        print(pd.DataFrame(wins).to_string(index=False))
    else:
        print("(none -- patched did not beat stock on MASE/WQL/CRPS on any dataset)")

    print("\n=== CALIBRATION WINS for patched (closer to nominal 80% coverage) ===")
    if cal_wins:
        print(pd.DataFrame(cal_wins).to_string(index=False))
    else:
        print("(none -- patched calibration is no better than stock on any dataset)")
"""))


# --------------------------------------------------------------------------
# 9. So, more compute or no?
# --------------------------------------------------------------------------
CELLS.append(md(r"""
    ## 9. So, more compute or no?

    Read the cells above and ask:

    - **At least one dataset where patched wins on a real metric by >5%?**
      That's a hint. Worth the architectural ablation step (TASK.md / ABLATION.md).
    - **Calibration consistently better than stock?** That's the parametric
      head's theoretical advantage actually showing up. Strongest signal
      for resource allocation, since calibration is what stock chronos
      *should* be worst at and what nobody seems to have a great solution
      to in the time-series literature.
    - **Patched competitive on tail quantiles (0.1, 0.9) even when worse
      on the median?** Same calibration story.
    - **Patched is uniformly worse on everything by 5-10%, no calibration
      improvement?** That's a "no signal" outcome at this scale. Two
      reads:
        1. The architecture really doesn't help. Don't spend more compute.
        2. The MLP head needs much more training than 4000 steps to be
           competitive at all, and we can't tell from this scale. Ablation
           step (20-30k steps on the 5090) before committing to the
           production run.

    Either way, the result is genuinely informative -- a "no signal"
    outcome at this scale is *also* worth knowing, because it constrains
    what next experiment to run.

    What this notebook does NOT show, even in the best case:

    - This is one fine-tune seed on a tiny KernelSynth-only corpus. The
      published Chronos recipe trains on TSMixup+KernelSynth at 0.9/0.1
      mix for 200k steps. A 4000-step fine-tune on KernelSynth alone is
      ~50000x less data exposure. Any signal here is noise-level
      compared to what a real run would produce.
    - One seed is one seed. The 2-sigma rule from TASK.md isn't meetable
      at this scale.
"""))


def write_notebook():
    for i, cell in enumerate(CELLS):
        cell.setdefault("id", f"cell-v2-{i:03d}")
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
