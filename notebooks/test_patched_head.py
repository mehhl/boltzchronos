"""Offline correctness check for T5ForMeanScalePatched.

No network needed: builds a T5 config from scratch, instantiates the patched
class with random weights, and verifies:

1. forward() produces log-probs that sum to ~1 in probability space
2. NLL loss is finite for random labels
3. gradients flow through the mean_scale_head
4. model.generate() runs without error and produces token IDs in the right range
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import T5Config, T5ForConditionalGeneration  # noqa: E402


# Inline copy of the patched class so this file is self-contained and stays in
# sync with the notebook by virtue of being unit-testable.
class T5ForMeanScalePatched(T5ForConditionalGeneration):
    def __init__(self, config, boundaries=None, n_special_tokens=2):
        super().__init__(config)
        d_vocab = config.vocab_size
        self.mean_scale_head = nn.Sequential(
            nn.Linear(d_vocab, 128), nn.ReLU(),
            nn.Linear(128, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )
        for layer in (0, 2, 4):
            nn.init.xavier_uniform_(self.mean_scale_head[layer].weight)
            nn.init.zeros_(self.mean_scale_head[layer].bias)
        if boundaries is None:
            n_bin_edges = d_vocab - n_special_tokens
            boundaries = torch.linspace(-15.0, 15.0, n_bin_edges)
            boundaries[0], boundaries[-1] = -1e20, 1e20
        self.register_buffer("boundaries", boundaries, persistent=False)
        n_init = d_vocab - (boundaries.numel() - 1)
        self.register_buffer(
            "init_log_probs", torch.full((n_init,), -1e9), persistent=False,
        )

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


def main():
    torch.manual_seed(0)

    # Tiny T5 config: just enough to spin up an encoder-decoder with our
    # 4096-token vocab. Don't bother trying to match T5-tiny dimensions.
    cfg = T5Config(
        vocab_size=4096,
        d_model=64,
        d_ff=128,
        d_kv=16,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )

    model = T5ForMeanScalePatched(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params, vocab={cfg.vocab_size}")

    # 1. forward + log-prob normalisation
    B, T_in, T_out = 2, 8, 4
    input_ids      = torch.randint(2, cfg.vocab_size, (B, T_in))
    attention_mask = torch.ones_like(input_ids)
    decoder_inputs = torch.randint(2, cfg.vocab_size, (B, T_out))
    labels         = torch.randint(2, cfg.vocab_size, (B, T_out))

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_inputs,
        labels=labels,
    )
    log_probs = out["logits"]
    probs = log_probs.exp()
    sum_probs = probs.sum(dim=-1)
    print(f"log_probs shape: {tuple(log_probs.shape)}")
    print(f"probs sum (per position): min={sum_probs.min().item():.6f} "
          f"max={sum_probs.max().item():.6f} mean={sum_probs.mean().item():.6f}")
    assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=2e-3), \
        "log-prob distribution does not sum to ~1"
    assert (probs >= 0).all() and (probs <= 1.0001).all(), "probs out of [0,1]"

    # 2. loss is finite
    loss = out["loss"]
    print(f"loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"

    # 3. gradients flow through mean_scale_head
    loss.backward()
    g0 = model.mean_scale_head[0].weight.grad
    g4 = model.mean_scale_head[4].weight.grad
    assert g0 is not None and g0.abs().sum() > 0, "no gradient on first MLP layer"
    assert g4 is not None and g4.abs().sum() > 0, "no gradient on last MLP layer"
    print(f"grad norms: layer0={g0.norm().item():.4f} layer4={g4.norm().item():.4f}")

    # 4. generate() works
    model.eval()
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        min_new_tokens=4,
        max_new_tokens=4,
        num_return_sequences=3,
        eos_token_id=cfg.eos_token_id,
        pad_token_id=cfg.pad_token_id,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    print(f"generate output shape: {tuple(gen.shape)}, "
          f"id range [{gen.min().item()}, {gen.max().item()}]")
    assert gen.min().item() >= 0 and gen.max().item() < cfg.vocab_size

    # 5. compare against the original T5ForMeanScale-style "return probs" bug
    # Verify that a softmax-of-probs (the buggy path) really does flatten the
    # distribution the way the notebook claims.
    mu = torch.tensor([0.0])
    sigma = torch.tensor([1.0])
    lp = model._censored_gaussian_logprobs(mu, sigma)[0]
    p = lp.exp()
    real_top5 = torch.topk(p, 5).values
    bug_softmax = torch.softmax(p, dim=-1)
    bug_top5 = torch.topk(bug_softmax, 5).values
    print("real top-5 probs:           ", [f"{x:.4f}" for x in real_top5.tolist()])
    print("buggy softmax(probs) top-5: ", [f"{x:.6f}" for x in bug_top5.tolist()])
    print("(if buggy distribution is much flatter, the notebook bug story holds)")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
