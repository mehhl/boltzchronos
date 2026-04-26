# Task: produce a defensible answer to "is boltzchronos an improvement on Chronos?"

This file is the spec for the next agent (human or otherwise) to pick up where
this branch leaves off. The branch already contains a runnable notebook
(`notebooks/boltzchronos_eval.ipynb`), a patched `T5ForMeanScale` that fixes
the inference-path bug, and an offline correctness check
(`notebooks/test_patched_head.py`). What remains is the actual experiment.

## Read these first, in this order

Roughly an hour of reading. Don't skip &mdash; the experiment is small once
you have the context, but the failure modes are subtle.

1. **This file.** All of it, end to end. Re-read "Success criteria" twice;
   the falsification rules are the part that's easy to bend by accident.
2. **`notebooks/boltzchronos_eval.ipynb`** &mdash; especially the markdown
   cell titled "The inference-path bug" and the cell defining
   `T5ForMeanScalePatched`. That's the single most important code change in
   this branch and the reason the original cloud run looked broken. The cells
   after it (KernelSynth gen, fine-tune, eval) are a working but tiny version
   of the experiment you're scaling up.
3. **`notebooks/test_patched_head.py`** &mdash; runs offline, takes ~5 seconds,
   verifies the patched head's math (log-probs sum to 1, gradients flow,
   `generate()` works) and demonstrates the bug numerically. Run it before
   touching the class, run it again after any change. Treat it as the
   regression test.
4. **`scripts/training/train.py:58-176`** &mdash; the original `T5ForMeanScale`
   class. Read it side-by-side with the patched version in the notebook so
   you understand exactly what changed and why. Notice the inline `print` on
   the loss (line 168), the hardcoded `cuda` (lines 70, 74, 92-94), the
   `init_probs` literal `[0,0,0]`, and the `outputs["logits"] = probs` at
   line 175. None of these are individually fatal but together they make the
   training dynamics opaque.
5. **`scripts/training/train.py:636-868`** &mdash; the training entry point.
   This is what you'll actually run for the production experiment, after
   swapping `T5ForMeanScale` for the patched class. Pay attention to the
   `CustomTrainer.compute_loss` (line 808) and how `boundaries` is plumbed
   from the tokenizer to the model (line 779).
6. **`scripts/evaluation/evaluate.py`** &mdash; the eval harness. Correct as-is;
   you'll add CRPS computation alongside the existing MASE/WQL and dump
   eval-time logs.
7. **`scripts/evaluation/configs/zero-shot.yaml`** and `in-domain.yaml` &mdash;
   the dataset lists. Note the four commented-out datasets in `zero-shot.yaml`
   (`m4_quarterly`, `m4_yearly`, `dominick`, `m5`); you'll uncomment them.
8. **`scripts/evaluation/results/zero-shot.csv`** &mdash; the *only* result
   row from the original cloud run: MASE 71.08, WQL 22.48 on `monash_traffic`
   vs. seasonal-naive's MASE 1.08, WQL 0.36. ~70x worse than the naive
   baseline. The owner of this repo recalled the run as "kinda successful"
   but the artifact disagrees. Most likely cause is the inference-path bug
   above. Don't be misled by this CSV; it is not a baseline to beat.
9. **`scripts/training/configs/chronos-t5-tiny.yaml`** &mdash; the training
   config that most closely matches the published recipe. Start your
   production run from a copy of this.
10. **The Chronos paper** ([arXiv:2403.07815](https://arxiv.org/abs/2403.07815)),
    sections 3 and 5. You don't need the full paper, just enough to know what
    the published numbers are on the 27-dataset zero-shot benchmark and how
    they aggregate. Section 5.5 ("Results") has the table you're trying to
    reproduce a row of.

You can skip:

- The rest of `src/chronos/chronos.py` (unchanged from upstream).
- The `test/` directory (a dummy random-init checkpoint, useful for unit
  tests but not for this work).
- Most of the `scripts/training/configs/debug/` files (per-dataset debug
  configs from a previous fork; not relevant here).

## Goal

Decide whether the censored-Gaussian output head ("Boltzmann distribution over
the existing token bins") meaningfully improves on stock Chronos when given
matched training compute. Produce numbers that would survive a hostile reading
&mdash; not a one-dataset accident, not an unintentional data leak, not a
fine-tune of a model that was already pretrained on the same corpus.

## Success criteria

The experiment is "complete" when, on the full 27-dataset zero-shot benchmark
from `scripts/evaluation/configs/zero-shot.yaml`, you can defend exactly one of
the following claims with the artifacts listed below.

1. **Win**: boltzchronos's geometric-mean **relative WQL** (vs. seasonal-naive
   per dataset, then geometric mean across datasets) is **strictly lower** than
   stock Chronos's at the same model size and training-token budget, with the
   gap holding across **at least 3 independent seeds** for both arms.
   "Strictly lower" means the seed-mean gap is at least 2x the larger of the
   two seed standard deviations. WQL is the headline metric from the Chronos
   paper; pick it as the tiebreaker.

2. **Calibration win, point-accuracy tie**: aggregated MASE within 5 % of
   stock Chronos AND aggregated CRPS strictly lower with the same 3-seed +
   2-sigma rule. This outcome is plausible because a parametric head's
   theoretical advantage is calibration, not point accuracy. Worth shipping.

3. **No improvement**: 3-seed runs of both arms produce overlapping
   distributions on every aggregate metric. Document the negative result and
   stop &mdash; do not chase it with more compute.

Anything that doesn't fit these three should be presented as inconclusive,
and the next experiment proposed (architectural change, more data, more
seeds, etc.).

### What does *not* count

- A single seed beating stock Chronos on one or two datasets.
- A fine-tune of `amazon/chronos-t5-tiny` (or any other published checkpoint)
  with the patched head. That model has already seen the eval-adjacent
  pretraining corpus; the comparison is not apples-to-apples.
- An eval where the seasonal-naive baseline numbers don't match the published
  Chronos paper to within a few percent on each dataset (a smoke test that the
  eval harness is correct).
- Numbers reported only as a screenshot of the notebook output. The CSVs go in
  the repo (see "Artifacts" below).

## "Production-size" numerically

Match the published Chronos-T5-tiny training recipe as closely as possible:

| Setting               | Value                                                |
| --------------------- | ---------------------------------------------------- |
| Model                 | `google/t5-efficient-tiny` random-init               |
| Vocab                 | 4096 tokens, 2 special                               |
| Context length        | 512                                                  |
| Prediction length     | 64                                                   |
| Per-device batch size | 32                                                   |
| Effective batch       | 256 (8 GPUs) or grad-accumulate to 256 on fewer GPUs |
| Optimizer             | `adamw_torch_fused`                                  |
| LR                    | 1e-3, linear decay                                   |
| Steps                 | 200_000                                              |
| Training data         | TSMixup + KernelSynth, 0.9 / 0.1 mix                 |
| Eval                  | full 27-dataset zero-shot config (see below)         |
| Seeds                 | 3, drawn from `[0, 1, 2]` for reproducibility        |

If you cannot afford 200k steps for two arms x three seeds, the minimum
acceptable scale is:

- 50_000 steps per run
- effective batch 128
- 2 seeds per arm

Anything below that is exploratory; report it as such and do not claim a
"win" or "no improvement" finding. Smaller runs are useful for the
architectural ablations in the next section, not for the headline result.

### Re-enable the commented-out datasets

`scripts/evaluation/configs/zero-shot.yaml` has `m4_quarterly`, `m4_yearly`,
`dominick`, and `m5` commented out. The published Chronos eval is on 27
datasets; this fork's config is on 23. Uncomment them so the comparison is
against the actual paper benchmark. Same for in-domain if running it.

## Artifacts the next agent must produce

All artifacts go in a `results/` directory at the repo root, on a results
branch (see "Where things go" below). File names use the pattern
`{benchmark}-{model}-seed{N}.csv` so multi-seed aggregation is mechanical.

Required:

- `results/zero-shot-stock-tiny-seed{0,1,2}.csv` &mdash; per-task MASE / WQL /
  CRPS for stock Chronos-T5-tiny re-run by you. Do not rely on the published
  numbers; the version of `transformers` / `gluonts` / `datasets` matters and
  you need a baseline collected with the same harness as the patched run.
- `results/zero-shot-boltz-tiny-seed{0,1,2}.csv` &mdash; same for the patched
  boltzchronos-tiny.
- `results/aggregate.csv` &mdash; one row per (model, seed) with the
  geometric-mean relative MASE, WQL, CRPS over the 27 datasets and the per-arm
  mean / std across seeds. Also include the seasonal-naive numbers used in the
  ratio.
- `results/training_logs/{model}-seed{N}/` &mdash; the HuggingFace
  `Trainer` output dir containing `trainer_state.json` and the TensorBoard
  events file. Optimization curves matter for sanity-checking that loss
  actually decreased and that the two arms had comparable training dynamics.
- `results/configs/{model}-seed{N}.yaml` &mdash; the exact training config used,
  including the git SHA of the repo and the package versions
  (`pip freeze > results/configs/{model}-seed{N}.pipfreeze.txt`).
- `results/eval_logs/{model}-seed{N}.log` &mdash; stdout/stderr from the eval
  run. Useful for spotting silent failures (skipped datasets, NaN forecasts,
  etc.) without re-running.

Required write-up:

- `README.md` &mdash; **the rewritten opener**. The current README is a
  verbatim copy of upstream Chronos's, which is misleading because this repo
  is not Chronos. Rewrite the top of the README to:
  - state in two sentences what boltzchronos is and how it differs from
    upstream Chronos (one-sentence math summary, link to the patched class);
  - state the headline result from `results/aggregate.csv` (one row of
    numbers; do not bury it in prose);
  - state the verdict ("win on calibration", "no improvement", etc.) and
    point at the artifacts.
  - move the inherited Chronos-paper material below a "## Upstream Chronos"
    section, kept verbatim, so attribution is preserved. Update
    `CITATION.cff` to reflect dual-citing this fork and the upstream paper.

Optional but appreciated:

- `results/per_dataset_plot.{png,pdf}` &mdash; per-dataset relative WQL bar
  chart, both arms overlaid. Makes "where does it win and where does it lose"
  obvious at a glance.
- `results/ablations.csv` &mdash; if you ran any of the architectural variants
  (head-from-hidden-states, mixture-of-K-Gaussians, learnable boundaries),
  one row per ablation against the same baseline.

## Where things go

Use a dedicated results branch so this branch stays focused on code. Suggested
naming:

```
results/boltz-tiny-v1
```

Branch off the current head of `claude/review-boltzmann-chronos-XB7Cj`. Commit
the artifacts there and push. Open a PR into `main` with a brief summary
(headline numbers + verdict). Do **not** force-push the results branch &mdash;
if you re-run experiments, append to it with a new commit so prior runs remain
inspectable.

If you also produced trained checkpoints worth sharing, do not commit binaries
to git. Push them to a Hugging Face model repo under your account
(e.g. `<you>/boltzchronos-t5-tiny-v1`) and put the link in the results branch
README.

## Suggested order of work

The cheap experiments inform the expensive one. Don't burn 200k steps on the
wrong architecture.

1. **Reproduce the baseline (cheap, hours).** Run `evaluate.py` against stock
   `amazon/chronos-t5-tiny` on the full 27-dataset zero-shot config with the
   harness pinned to whatever package versions you'll use for the patched run.
   Confirm the numbers are within a few percent of the paper's; if not, fix the
   harness before going further.
2. **End-to-end smoke (cheap, ~hour on one GPU).** Run the notebook on Colab
   to confirm the patched class trains and evals end to end. The output is not
   the experiment, just a sanity check on the plumbing.
3. **Architectural ablation (medium, ~day on one GPU).** At 20k-30k-step
   scale, compare:
   - `T5ForMeanScalePatched` as-is (head reads from `lm_head` logits),
   - head reading from `decoder_hidden_states` instead,
   - mixture of K=5 Gaussians.
   Pick the winner on aggregated WQL+CRPS over a 5-dataset subset of zero-shot
   (e.g. `monash_tourism_yearly`, `exchange_rate`, `monash_covid_deaths`,
   `nn5_weekly`, `monash_m1_yearly`). Use this winner for step 4.
4. **Production run (expensive, ~day on 8xA100 or equivalent).** 200k steps,
   3 seeds, both arms, full 27-dataset eval. Produce all artifacts above.

## Pointers to existing code

- Patched output head: `notebooks/boltzchronos_eval.ipynb` (cell with
  `T5ForMeanScalePatched`) and `notebooks/test_patched_head.py`.
- Original (buggy) head: `scripts/training/train.py` lines 58-176.
- Eval harness: `scripts/evaluation/evaluate.py`. Already correct; the only
  thing to add for production is dumping the eval log and writing CRPS
  alongside MASE/WQL.
- Training entry point: `scripts/training/train.py`. To use the patched head
  in production, replace the `T5ForMeanScale` import with the patched class
  (or copy the patched class into `train.py` and remove the original); pass
  the tokenizer's real `boundaries` through, as already done at line 779.
- Training configs: `scripts/training/configs/chronos-t5-tiny.yaml` is the
  closest match to the published recipe.

## Open questions worth resolving as you go

These do not block the headline experiment but inform follow-ups:

- The MLP head reads from the 4096-d `lm_head` output rather than from the
  256-d `decoder_hidden_states`. That's wasteful and probably hurts; the
  ablation in step 3 should answer it.
- `boundaries` is a `requires_grad=False` buffer. The original
  `train.py:840` reads `model.boundaries` post-training as if expecting them
  to have changed. Decide whether learnable boundaries are worth the
  complication; if yes, register as a parameter, not a buffer.
- Sampling at inference: with the patched class, `predict()` calls
  `model.generate(do_sample=True, top_k=50)`. Top-k on a peaked Gaussian
  distribution may be redundant; consider switching inference to direct
  sampling from the censored Gaussian rather than going through HF generate.
  Test this *after* the production run so it doesn't perturb the training/
  eval comparison.
