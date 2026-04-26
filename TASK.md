# How to run the boltzchronos evals

This branch contains a Colab notebook and three small CPU-runnable scripts for
poking at boltzchronos &mdash; the censored-Gaussian-output-head variant of
Chronos &mdash; and comparing it against stock `amazon/chronos-t5-tiny`. None
of these are the "real" production experiment (see "Going bigger" at the
bottom); they're sized to fit on a free Colab T4 or a laptop CPU.

## What boltzchronos is

Stock Chronos: T5 with a 4096-way categorical softmax over bin tokens, trained
with cross-entropy.

boltzchronos: same T5 trunk, but the categorical logits feed an MLP that
outputs `(mu, sigma)` of a Gaussian. The Gaussian is integrated between the
tokenizer's bin boundaries to produce a probability distribution over the
same 4096 tokens. Loss is NLL on those bin probabilities. See
`notebooks/boltzchronos_eval.ipynb` (cell defining `T5ForMeanScalePatched`)
or `scripts/training/train.py:58-176` for the original.

The original `T5ForMeanScale.forward` returned *probabilities* in
`outputs.logits`, but HuggingFace's sampler softmaxes those again, which
washes out the distribution. `T5ForMeanScalePatched` returns log-probabilities
instead. That's the single most consequential change in this branch and is
why the prior cloud run on `monash_traffic` looked broken
(`scripts/evaluation/results/zero-shot.csv`: MASE 71 vs seasonal-naive's 1.08).

## What to run

Each script is self-contained, prints a clear summary, and writes a CSV under
`scripts/evaluation/results/`.

### 1. `notebooks/test_patched_head.py` &mdash; offline correctness check

No network, no datasets, no checkpoint downloads. Builds a tiny T5 from
scratch, instantiates `T5ForMeanScalePatched`, and checks:

- forward pass produces log-probs that exponentiate to a valid distribution,
- NLL loss is finite,
- gradients flow through `mean_scale_head`,
- `model.generate()` runs and produces token IDs in range,
- the original "return probs" bug actually flattens the distribution.

Runs in ~5 seconds. Use it as a regression test before and after touching
the patched class.

```bash
python notebooks/test_patched_head.py
```

### 2. `notebooks/sanity_eval.py` &mdash; stock-only baseline on one dataset

Downloads `amazon/chronos-t5-tiny` (~30 MB) and `monash_tourism_yearly` from
Hugging Face, runs the eval harness, prints MASE / WQL, and writes
`scripts/evaluation/results/sanity-cpu-baseline.csv`. Confirms your
`transformers` / `gluonts` / `datasets` versions still work end-to-end
before touching anything else. ~5 minutes on a 2-core CPU.

```bash
python notebooks/sanity_eval.py
```

The seasonal-naive reference for this dataset is roughly `MASE = 3.55,
WQL = 0.21`; stock chronos-t5-tiny should be in the same ballpark or better.

### 3. `notebooks/mini_eval.py` &mdash; mini stock-vs-patched A/B

End-to-end CPU comparison. Evals stock chronos-t5-tiny, generates a tiny
KernelSynth dataset, briefly fine-tunes `T5ForMeanScalePatched` from the
stock checkpoint, evals the patched model on the same task, and writes both
rows (with MASE, WQL, CRPS) to
`scripts/evaluation/results/mini-eval.csv`.

```bash
python notebooks/mini_eval.py
```

Defaults are aggressive (200 fine-tune steps, 200 synthetic series, batch 2)
so the run finishes in ~15-30 min on a laptop CPU. **At this scale the
fine-tune is heavily under-trained**; the patched numbers will look bad. The
script's job is to verify (a) the inference path works, (b) the eval
harness handles both models, (c) loss decreases. For a real comparison use
the Colab notebook below.

Tunables (env vars):

| Var                  | Default                  | Notes                         |
| -------------------- | ------------------------ | ----------------------------- |
| `MINI_EVAL_STEPS`    | 200                      | fine-tune steps               |
| `MINI_EVAL_N_SYNTH`  | 200                      | synthetic series              |
| `MINI_EVAL_DATASET`  | `monash_tourism_yearly`  | eval dataset                  |
| `MINI_EVAL_BATCH`    | 2                        | per-device train batch        |
| `MINI_EVAL_NO_TRAIN` | unset                    | if set, skip fine-tune        |

### 4. `notebooks/boltzchronos_eval.ipynb` &mdash; Colab notebook

The same comparison as `mini_eval.py` but sized for a Colab T4: 3000-series
KernelSynth and 2000 fine-tune steps (~50 min on a T4), with proper sample
forecasts and a side-by-side summary at the end.

Open it directly in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehhl/boltzchronos/blob/claude/complete-task-md-v8TeV/notebooks/boltzchronos_eval.ipynb)

(or open `notebooks/boltzchronos_eval.ipynb` from the GitHub UI and click the
badge at the top.)

The notebook is generated from `notebooks/build_notebook.py`. If you edit it,
edit the `CELLS` list there and re-run the script &mdash; do not edit the
`.ipynb` JSON by hand.

## What a "good" mini result looks like

None of these will tell you whether boltzchronos beats stock Chronos on the
27-dataset zero-shot benchmark. The mini eval *will* tell you:

1. **Inference plumbing is intact.** Patched MASE on `monash_tourism_yearly`
   is finite and within ~10x of stock. (At 200 fine-tune steps it can easily
   be 2-5x worse than stock; that's expected, not a regression.)
2. **Loss is going down.** The trainer prints NLL every ~20 steps. Should
   decrease monotonically; if it plateaus immediately, the MLP head can't
   learn from the `lm_head` logits and you should switch to projecting from
   the decoder hidden states (one of the open questions below).
3. **CRPS is reported.** It isn't in the original `evaluate.py`; the mini
   eval and the notebook both compute it from sample forecasts. For a
   parametric calibrated head, CRPS is the metric most likely to win even
   when MASE doesn't.

## Open questions worth poking at

These do not block running the scripts above; they're follow-ups suggested
by the patched class.

- The MLP head reads from the 4096-d `lm_head` output rather than the
  256-d `decoder_hidden_states`. Wasteful and probably hurts.
- `boundaries` is a `requires_grad=False` buffer. The original
  `train.py:840` reads `model.boundaries` post-training as if expecting
  it to have changed. Register it as a parameter to test learnable bins.
- Sampling at inference goes through HF `generate(do_sample=True, top_k=50)`.
  Top-k on a peaked Gaussian distribution is probably redundant; consider
  sampling directly from the censored Gaussian. Test *after* any comparison
  run so it doesn't perturb the train/eval baseline.

## Going bigger

The headline question &mdash; "is boltzchronos an improvement on Chronos?"
&mdash; is not answerable from any of the runs above. A defensible answer
needs the published Chronos training recipe (T5-tiny random-init, ~200k
steps on TSMixup + KernelSynth, batch 256, full 27-dataset zero-shot eval)
across both arms and at least 3 seeds each. That's roughly 3 days on
8x A100. The previous spec for that experiment lives in this file's git
history at commit `ed980c2`; resurrect it if you have the compute.
