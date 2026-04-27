"""Mini CPU end-to-end comparison: stock chronos-t5-tiny vs patched boltzchronos.

End-to-end version of `sanity_eval.py`: in addition to the stock baseline, this
also generates a tiny KernelSynth dataset, briefly fine-tunes
`T5ForMeanScalePatched` from the stock checkpoint, evals it on the same task,
and writes both rows to `scripts/evaluation/results/mini-eval.csv` with MASE,
WQL, and CRPS.

Defaults are sized to finish in ~15-30 min on a 2-core laptop CPU. With the
defaults, 200 fine-tune steps is *not* enough to give a fair comparison, but
it does verify (1) the inference path works end-to-end, (2) the eval harness
runs both models without exploding, and (3) loss is going down. For a fairer
comparison, run the Colab notebook with a GPU and crank `MAX_STEPS`.

Tunables (env vars):
  MINI_EVAL_STEPS    fine-tune steps (default 200)
  MINI_EVAL_N_SYNTH  number of synthetic series for fine-tuning (default 200)
  MINI_EVAL_DATASET  HF dataset name for eval (default monash_tourism_yearly)
  MINI_EVAL_BATCH    per-device train batch size (default 2)
  MINI_EVAL_NO_TRAIN if set, skip the fine-tune and eval the patched class
                     with the random MLP head. Useful for plumbing-only checks.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gluonts.dataset.arrow import ArrowWriter
from gluonts.dataset.common import FileDataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import Cyclic, Filter, Map, batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    FilterTransformation,
    InstanceSplitter,
)
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from chronos import ChronosConfig, ChronosPipeline  # noqa: E402
from chronos.chronos import ChronosModel  # noqa: E402
from chronos.chronos import ChronosPipeline as ChronosPipelineCls  # noqa: E402

OFFSET_TO_PERIOD = {
    "MS": "M", "ME": "M", "M": "M",
    "QS": "Q", "QE": "Q", "Q": "Q",
    "YS": "Y", "YE": "Y", "Y": "Y",
    "W": "W", "D": "D", "h": "h", "min": "min", "s": "s",
}


def to_period_freq(freq):
    """Map a pandas DateOffset alias to a Period-compatible alias.

    pd.infer_freq returns offset aliases like 'YE-DEC' (Year End, December
    anchor). pd.Period only accepts 'Y-DEC'. Strip the offset suffix off the
    base part while preserving any '-ANCHOR' tail.
    """
    if freq is None or freq in OFFSET_TO_PERIOD:
        return OFFSET_TO_PERIOD.get(freq, freq)
    base, sep, anchor = freq.partition("-")
    if base in OFFSET_TO_PERIOD:
        period_base = OFFSET_TO_PERIOD[base]
        return f"{period_base}-{anchor}" if anchor else period_base
    return freq

DATASETS = {
    "monash_tourism_yearly": dict(
        name="monash_tourism_yearly",
        hf_repo="autogluon/chronos_datasets",
        offset=-4,
        prediction_length=4,
        num_rolls=1,
    ),
}


# ---------------------------------------------------------------------------
# Patched class. Inline copy of the notebook's T5ForMeanScalePatched so this
# script is self-contained. Stays in sync with the notebook by virtue of
# test_patched_head.py running against the same definition.
# ---------------------------------------------------------------------------
class T5ForMeanScalePatched(T5ForConditionalGeneration):
    def __init__(self, config, boundaries=None, n_special_tokens=2):
        super().__init__(config)
        d_vocab = config.vocab_size
        self.mean_scale_head = nn.Sequential(
            nn.Linear(d_vocab, 128), nn.ReLU(),
            nn.Linear(128, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )
        # Tag the MLP Linears so _init_weights can find them. Direct
        # nn.init calls here would no-op under the meta-tensor path that
        # from_pretrained uses, leaving the layers with uninitialised
        # memory after materialisation.
        for layer in self.mean_scale_head:
            if isinstance(layer, nn.Linear):
                layer._is_mean_scale_head = True
        self._init_mean_scale_head()
        if boundaries is None:
            # Match the chronos MeanScaleUniformBins geometry: with default
            # vocab=4096 and 2 special tokens, n_bin_edges=4094 (so n_init=3,
            # matching the original train.py's init_probs=[0,0,0]).
            n_bin_edges = d_vocab - n_special_tokens
            boundaries = torch.zeros(n_bin_edges)
        # persistent=False so save_pretrained doesn't pickle the buffer; we
        # always override from the tokenizer at load time anyway, and keeping
        # them in the state_dict creates shape-mismatch traps on reload.
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
        # T5's _init_weights leaves generic nn.Linear untouched; under
        # from_pretrained's meta-tensor path that produces uninitialised
        # memory in mean_scale_head. Re-init any Linear we marked.
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


# ---------------------------------------------------------------------------
# Eval helpers (mirror the notebook).
# ---------------------------------------------------------------------------
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


def run_full_eval(pipeline, dataset_cfg, num_samples=20, batch_size=8):
    pred_len = dataset_cfg["prediction_length"]
    td = make_test_data(**dataset_cfg)
    all_samples = []
    for batch in tqdm(batcher(td.input, batch_size=batch_size), desc="predict"):
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

    td4 = make_test_data(**dataset_cfg)
    df = evaluate_forecasts(
        forecasts, test_data=td4,
        metrics=[MASE(), MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1))],
        batch_size=5000,
    ).reset_index(drop=True)
    row = df.to_dict(orient="records")[0]
    crps = crps_from_samples(forecast_samples, np.stack(targets))
    return {
        "MASE": float(row["MASE[0.5]"]),
        "WQL": float(row["mean_weighted_sum_quantile_loss"]),
        "CRPS": crps,
    }


# ---------------------------------------------------------------------------
# Tiny KernelSynth + training dataset.
# ---------------------------------------------------------------------------
def load_kernel_synth_module():
    spec = importlib.util.spec_from_file_location(
        "kernel_synth", REPO_ROOT / "scripts/kernel-synth.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kernel_synth"] = mod
    spec.loader.exec_module(mod)
    return mod


def gen_kernelsynth(out_path: Path, n_synth: int, max_kernels: int = 4):
    if out_path.exists():
        print(f"  {out_path} exists ({out_path.stat().st_size / 1e6:.1f} MB) - reusing")
        return
    ks = load_kernel_synth_module()
    # Serial generation: kernel-synth.py is loaded via importlib from a
    # hyphenated filename, so a joblib loky worker can't re-import it.
    # 200-series default finishes in ~1 min on a 2-core CPU; bump
    # MINI_EVAL_N_SYNTH only when paired with a faster box.
    series = [
        ks.generate_time_series(max_kernels=max_kernels)
        for _ in tqdm(range(n_synth), desc="kernel-synth")
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ArrowWriter(compression="lz4").write_to_file(series, path=out_path)
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


class TinyChronosDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, context_length=128,
                 prediction_length=8, min_past=8, drop_prob=0.2):
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
            condition=lambda e: (~np.isnan(e["past_target"])).sum() > 0,
        )

    def _to_hf(self, entry):
        past = torch.tensor(entry["past_target"]).unsqueeze(0)
        future = torch.tensor(entry["future_target"]).unsqueeze(0)
        input_ids, attn_mask, scale = self.tokenizer.context_input_transform(past)
        labels, labels_mask = self.tokenizer.label_input_transform(future, scale)
        labels[labels_mask == 0] = -100
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attn_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def _preprocess(self, entry):
        target = np.asarray(entry["target"], dtype=np.float32)
        if self.drop_prob > 0:
            p = np.random.uniform(0.0, self.drop_prob)
            mask = np.random.choice([True, False], size=len(target), p=[p, 1 - p])
            target = target.copy()
            target[mask] = np.nan
        return {"start": entry["start"], "target": target}

    def __iter__(self):
        data = Cyclic(self.dataset)
        data = Map(self._preprocess, data)
        data = self._splitter().apply(data, is_train=True)
        for entry in data:
            yield self._to_hf(entry)


# ---------------------------------------------------------------------------
# Pipeline construction for the patched class.
# ---------------------------------------------------------------------------
def build_patched_pipeline(checkpoint, device, dtype):
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


def fine_tune_patched(out_dir: Path, synth_path: Path, max_steps: int,
                      batch_size: int, ctx_len: int, pred_len: int) -> Path:
    chronos_cfg_dict = AutoConfig.from_pretrained("amazon/chronos-t5-tiny").chronos_config
    chronos_cfg = ChronosConfig(**chronos_cfg_dict)
    tokenizer = chronos_cfg.create_tokenizer()

    train_ds = TinyChronosDataset(
        file_path=str(synth_path),
        tokenizer=tokenizer,
        context_length=ctx_len,
        prediction_length=pred_len,
        min_past=pred_len,
        drop_prob=0.2,
    )

    model = T5ForMeanScalePatched.from_pretrained(
        "amazon/chronos-t5-tiny", torch_dtype=torch.float32,
    )
    model.boundaries = tokenizer.boundaries.float()
    model.config.chronos_config = chronos_cfg.__dict__

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **_):
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        learning_rate=5e-4,
        max_steps=max_steps,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=max(1, max_steps // 20),
        report_to=[],
        bf16=False,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        torch_compile=False,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = CustomTrainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()
    ckpt = out_dir / "checkpoint-final"
    model.save_pretrained(ckpt)
    return ckpt


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    steps = int(os.environ.get("MINI_EVAL_STEPS", "200"))
    n_synth = int(os.environ.get("MINI_EVAL_N_SYNTH", "200"))
    dataset_key = os.environ.get("MINI_EVAL_DATASET", "monash_tourism_yearly")
    batch = int(os.environ.get("MINI_EVAL_BATCH", "2"))
    skip_train = os.environ.get("MINI_EVAL_NO_TRAIN", "") not in ("", "0", "false", "False")

    if dataset_key not in DATASETS:
        raise SystemExit(f"Unknown dataset {dataset_key!r}. Known: {list(DATASETS)}")
    dataset_cfg = DATASETS[dataset_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"device={device} dtype={dtype} steps={steps} n_synth={n_synth} "
          f"dataset={dataset_key} batch={batch} skip_train={skip_train}")

    rows = []

    # 1. stock baseline
    print("\n[1/3] eval stock chronos-t5-tiny")
    t0 = time.time()
    stock_pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny", device_map=device, torch_dtype=dtype,
    )
    stock_metrics = run_full_eval(stock_pipe, dataset_cfg, num_samples=20, batch_size=8)
    elapsed = time.time() - t0
    print(f"  stock: {stock_metrics} ({elapsed:.1f}s)")
    rows.append({
        "model": "stock-chronos-t5-tiny",
        "dataset": dataset_key,
        "train_steps": 0,
        "device": device,
        "elapsed_s": f"{elapsed:.1f}",
        **{k: f"{v:.6f}" for k, v in stock_metrics.items()},
    })
    del stock_pipe

    # 2. fine-tune patched (or skip)
    if skip_train:
        print("\n[2/3] MINI_EVAL_NO_TRAIN set, skipping fine-tune")
        ckpt = "amazon/chronos-t5-tiny"
        train_steps = 0
    else:
        print(f"\n[2/3] generate {n_synth}-series KernelSynth + fine-tune patched ({steps} steps)")
        synth_path = REPO_ROOT / "data/kernelsynth-mini.arrow"
        gen_kernelsynth(synth_path, n_synth=n_synth)
        out_dir = REPO_ROOT / "output/run-mini-patched"
        t0 = time.time()
        # pred_len must match chronos_config.prediction_length (=64 for the
        # pretrained chronos-t5-tiny tokenizer). The tokenizer's
        # label_input_transform asserts this, so we can't just dial it down
        # for CPU. ctx_len is still tunable.
        ckpt = fine_tune_patched(
            out_dir=out_dir,
            synth_path=synth_path,
            max_steps=steps,
            batch_size=batch,
            ctx_len=128,
            pred_len=64,
        )
        print(f"  fine-tune: {time.time() - t0:.1f}s -> {ckpt}")
        train_steps = steps

    # 3. patched eval
    print("\n[3/3] eval patched")
    t0 = time.time()
    patched_pipe = build_patched_pipeline(checkpoint=str(ckpt), device=device, dtype=dtype)
    patched_metrics = run_full_eval(patched_pipe, dataset_cfg, num_samples=20, batch_size=8)
    elapsed = time.time() - t0
    print(f"  patched: {patched_metrics} ({elapsed:.1f}s)")
    rows.append({
        "model": "boltzchronos-patched" + ("-untrained" if skip_train else ""),
        "dataset": dataset_key,
        "train_steps": train_steps,
        "device": device,
        "elapsed_s": f"{elapsed:.1f}",
        **{k: f"{v:.6f}" for k, v in patched_metrics.items()},
    })

    # 4. write CSV + summary
    out_csv = REPO_ROOT / "scripts/evaluation/results/mini-eval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "dataset", "MASE", "WQL", "CRPS", "train_steps", "device", "elapsed_s"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nwrote {out_csv}")

    print("\n=== summary ===")
    print(pd.DataFrame(rows)[cols].to_string(index=False))
    print(
        "\nNote: with the default MINI_EVAL_STEPS=200 and N_SYNTH=200 the "
        "patched model is heavily under-trained;\nthe comparison only verifies "
        "the inference path works and the eval runs. For a fairer\ncomparison, "
        "run notebooks/boltzchronos_eval.ipynb on a Colab GPU and crank MAX_STEPS."
    )


if __name__ == "__main__":
    main()
