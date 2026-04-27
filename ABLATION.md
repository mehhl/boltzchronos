# Architectural ablation: which head variant trains best?

This is the cheap experiment from `TASK.md` step 3. Before committing to the
full 200k-step production run, train and eval three variants of the
censored-Gaussian output head on a 5-dataset zero-shot subset to pick the
architectural winner.

If you don't care about the design, skip to [Run it](#run-it).

## Variants

All three live in `scripts/training/train.py` (selected by the
`--head-variant` flag) and produce the same kind of output (log-probs over
the 4096-token chronos vocab, integrated from a censored Gaussian over the
tokenizer's bin boundaries). The differences are upstream of that.

| Variant   | Reads from                    | Output head | MLP first layer       |
| --------- | ----------------------------- | ----------- | --------------------- |
| `lm_head` | `lm_head` logits (V=4096)     | (mu, sigma) | Linear(4096, 128)     |
| `hidden`  | last decoder hidden state     | (mu, sigma) | Linear(d_model, 128)  |
| `mixture` | last decoder hidden state     | K=5 mixture | Linear(d_model, 128)  |

`lm_head` is the original (now patched) baseline. `hidden` tests the
hypothesis from `TASK.md`'s "open questions" that reading from the 4096-d
`lm_head` is wasteful and probably hurts -- with T5-tiny `d_model=64`,
the MLP is 64x smaller on its first layer. `mixture` tests whether 5
Gaussians give meaningfully better calibration than 1.

`lm_head` and `hidden` produce models with very different parameter counts
(0.90M vs 0.39M for T5-tiny). `hidden` and `mixture` have nearly the same
count -- the comparison between them is purely about expressivity of the
output head.

## Run it

Prerequisite: you've staged TSMixup + KernelSynth as `.arrow` files
somewhere on your box. (KernelSynth can be regenerated with
`scripts/kernel-synth.py`; TSMixup is upstream Amazon's preprocessed
dataset.)

```bash
TRAINING_DATA=/data/tsmixup-data.arrow,/data/kernelsynth-data.arrow \
    bash scripts/run_ablation.sh
```

Defaults (override via env vars):

| Var               | Default                                                 |
| ----------------- | ------------------------------------------------------- |
| `TRAINING_DATA`   | (required)                                              |
| `SEED`            | `0`                                                     |
| `OUTPUT_BASE`     | `./output/ablation`                                     |
| `RESULTS_DIR`     | `./results/ablation`                                    |
| `TRAIN_CONFIG`    | `scripts/training/configs/chronos-t5-tiny-ablation.yaml`|
| `EVAL_CONFIG`     | `scripts/evaluation/configs/zero-shot-ablation.yaml`    |
| `MAX_STEPS`       | (uses the yaml's `max_steps: 25_000`)                   |

The runner trains all three variants sequentially, then evals each on the
5-dataset config, then aggregates into `results/ablation/aggregate-seed0.csv`
with geometric-mean MASE / WQL / CRPS per variant.

Skip-rebuild logic: if `output/ablation/<variant>-seed0/checkpoint-final/`
already exists, training for that variant is skipped and the existing
checkpoint is re-evaluated. Useful for iterating on the eval side.

## Estimated cost

On a single RTX 5090 (32 GB, sm_120, bf16):

- Per-variant training at 25k steps, batch 32, grad-accum 8 (effective 256),
  ctx 512, pred 64: roughly **2-3 hours**.
- Per-variant eval on the 5-dataset subset: roughly **5-15 minutes**.
- Three variants total: **6-10 hours wall-clock**.

If that's too long, halve `MAX_STEPS` to 12500 -- the loss curves stabilise
well before 25k, and the *relative* ordering between variants should
already be visible.

## Picking the winner

Look at `aggregate-seed0.csv`. The variant with the lowest geometric-mean
WQL+CRPS wins. If the gap is within run-to-run noise (rerun with `SEED=1`
to estimate noise), keep `lm_head` -- not because it's better but because
it's the published recipe and the production run is more interpretable
that way.

The winner is what you should use for the headline 200k-step run.

## Troubleshooting

**`from train import HEAD_VARIANTS`**: `evaluate.py` does this at
`--head-variant` time. If you renamed `train.py`, update
`_build_patched_pipeline` in `evaluate.py`.

**Loss NaN or stuck at 0**: re-run `python notebooks/test_patched_head.py`
to verify the offline math still works. If that passes, the bug is
upstream of the head -- check tokenizer boundaries, label masking, or
data path.

**OOM on the 5090**: reduce `per_device_train_batch_size` to 16 and double
`gradient_accumulation_steps` to 16 to keep the effective batch at 256.

**`xet` errors downloading from HF**: `export HF_HUB_DISABLE_XET=1` and
retry. We hit this on the CPU mini-eval too.
