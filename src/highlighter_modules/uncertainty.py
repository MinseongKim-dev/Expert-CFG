"""
Predictive Entropy (PE) computation for uncertainty-driven CFG control.

Implements Expert-CFG paper Eq. 4:
  PE(x) = -(1/N) * Σ_{i=1}^{N} Σ_{v} P(x_i=v | x_{<i}) * log P(x_i=v | x_{<i})

This length-normalised entropy measures model uncertainty over the generated sequence.
High PE → model is uncertain → expert guidance (CFG) is more beneficial.
"""

from __future__ import annotations

import torch


def compute_predictive_entropy(scores: list[torch.Tensor]) -> float:
    """Compute length-normalised predictive entropy from generation scores.

    Args:
        scores: List of per-step logit tensors returned by model.generate()
                with output_scores=True. Each tensor is [batch_size, vocab_size].
                We use index 0 (conditional branch of the CFG batch).

    Returns:
        Scalar PE value (nats). Returns 0.0 for empty sequences.
    """
    N = len(scores)
    if N == 0:
        return 0.0

    total_entropy = 0.0
    for step_scores in scores:
        # Use the conditional branch (index 0 of the CFG 2x batch)
        logits = step_scores[0].float()  # [vocab_size]
        probs = torch.softmax(logits, dim=-1)
        # H = -Σ p*log(p), clamp for numerical stability
        entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum().item()
        total_entropy += entropy

    return total_entropy / N
