"""Debug: run one batch through the patched model and print loss components."""
from __future__ import annotations

import os, sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

import torch
from transformers import AutoConfig
from chronos import ChronosConfig

from mini_eval import T5ForMeanScalePatched, TinyChronosDataset

cfg = AutoConfig.from_pretrained("amazon/chronos-t5-tiny")
chronos_cfg = ChronosConfig(**cfg.chronos_config)
tokenizer = chronos_cfg.create_tokenizer()
print("tokenizer.boundaries shape:", tokenizer.boundaries.shape)
print("tokenizer.boundaries[0:3]:", tokenizer.boundaries[:3].tolist())
print("tokenizer.boundaries[-3:]:", tokenizer.boundaries[-3:].tolist())

model = T5ForMeanScalePatched.from_pretrained(
    "amazon/chronos-t5-tiny", torch_dtype=torch.float32,
)
model.boundaries = tokenizer.boundaries.float()
print("model.boundaries shape:", model.boundaries.shape)
print("model.init_log_probs shape:", model.init_log_probs.shape)
print("model.config.vocab_size:", model.config.vocab_size)

for i, layer in enumerate(model.mean_scale_head):
    if hasattr(layer, "weight"):
        w = layer.weight
        b = layer.bias if hasattr(layer, "bias") and layer.bias is not None else None
        print(f"  mean_scale_head[{i}] {type(layer).__name__}: w shape={tuple(w.shape)} std={w.std().item():.4e} max_abs={w.abs().max().item():.4e}", end="")
        if b is not None:
            print(f" bias std={b.std().item():.4e} max_abs={b.abs().max().item():.4e}")
        else:
            print()

ds = TinyChronosDataset(
    file_path=str(REPO_ROOT / "data/kernelsynth-mini.arrow"),
    tokenizer=tokenizer,
    context_length=128,
    prediction_length=64,
    min_past=64,
    drop_prob=0.2,
)
it = iter(ds)
sample = next(it)
print("\nsample shapes:")
for k, v in sample.items():
    print(f"  {k}: {v.shape} dtype={v.dtype}")
print("labels[:20]:", sample["labels"][:20].tolist())
print("labels min/max:", sample["labels"].min().item(), sample["labels"].max().item())
print("labels==-100 count:", (sample["labels"] == -100).sum().item(), "/", sample["labels"].numel())
print("labels valid count:", (sample["labels"] != -100).sum().item())

# Run forward
batch = {k: v.unsqueeze(0) for k, v in sample.items()}
out = model(**batch)
print("\nforward output:")
print("  loss:", out["loss"])
print("  logits (log_probs) shape:", out["logits"].shape)
print("  logits dtype:", out["logits"].dtype)
print("  logits has nan?", torch.isnan(out["logits"]).any().item())
print("  logits has inf?", torch.isinf(out["logits"]).any().item())
print("  logits min/max:", out["logits"].min().item(), out["logits"].max().item())

# Inspect mu/sigma directly
with torch.no_grad():
    t5_out = model.get_encoder()(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    dec = model.get_decoder()(
        input_ids=model._shift_right(batch["labels"].masked_fill(batch["labels"] == -100, 0)),
        encoder_hidden_states=t5_out.last_hidden_state,
        encoder_attention_mask=batch["attention_mask"],
    )
    t5_logits = model.lm_head(dec.last_hidden_state)
    print("t5_logits stats: mean=%.4f std=%.4f min=%.4f max=%.4f abs_max=%.4f" % (
        t5_logits.mean().item(), t5_logits.std().item(),
        t5_logits.min().item(), t5_logits.max().item(),
        t5_logits.abs().max().item()))
    flat = t5_logits.reshape(-1, t5_logits.size(-1))
    # MLP step by step
    h0 = model.mean_scale_head[0](flat)
    print("mlp[0] (Linear 4096->128) out stats: max_abs=%.4e mean_abs=%.4e" % (h0.abs().max().item(), h0.abs().mean().item()))
    h1 = model.mean_scale_head[1](h0)  # ReLU
    h2 = model.mean_scale_head[2](h1)
    print("mlp[2] (Linear 128->16)   out stats: max_abs=%.4e mean_abs=%.4e" % (h2.abs().max().item(), h2.abs().mean().item()))
    h3 = model.mean_scale_head[3](h2)  # ReLU
    h4 = model.mean_scale_head[4](h3)
    print("mlp[4] (Linear 16->2)     out stats: max_abs=%.4e mean_abs=%.4e" % (h4.abs().max().item(), h4.abs().mean().item()))
    ms = h4
    mu = ms[:, 0]
    sigma_pre = ms[:, 1]
    sigma = torch.relu(ms[:, 1] - 1e-10) + 1e-10
    print("\nmu     stats: mean=%.6f std=%.6f min=%.6f max=%.6f" % (mu.mean().item(), mu.std().item(), mu.min().item(), mu.max().item()))
    print("sigma_pre stats: mean=%.6f std=%.6f min=%.6f max=%.6f" % (sigma_pre.mean().item(), sigma_pre.std().item(), sigma_pre.min().item(), sigma_pre.max().item()))
    print("sigma  stats: mean=%.6f std=%.6f min=%.6f max=%.6f" % (sigma.mean().item(), sigma.std().item(), sigma.min().item(), sigma.max().item()))

# Try backward
print("\nbackward:")
loss = out["loss"]
print("  loss.item():", loss.item())
print("  loss requires_grad?", loss.requires_grad)
loss.backward()
g = model.mean_scale_head[4].weight.grad
print("  mean_scale_head[4].weight.grad has nan?", torch.isnan(g).any().item())
print("  mean_scale_head[4].weight.grad has inf?", torch.isinf(g).any().item())
print("  mean_scale_head[4].weight.grad max abs:", g.abs().max().item())
