import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# (A) Proper MoE likelihood: observed-data negative log-likelihood
# ------------------------------------------------------------


def moe_nll(batch, moe_model):
    """
    Negative observed-data log-likelihood for a mixture-of-experts model:

        -log sum_k pi_k(x) p(y | x, k)

    This is the correct probabilistic MoE objective.
    """
    eps = moe_model.eps

    y = batch["phenotype"].float().view(-1)  # (N,)
    w = moe_model.gate_forward(batch).clamp_min(eps)  # (N, K)

    expert_predictions = moe_model._combined_linear_predictor(batch)  # (N, K)

    if moe_model.family == "gaussian":
        sigma_sq = moe_model.sigma2 if moe_model.sigma2 is not None else 1.0
        sigma_sq = torch.as_tensor(
            sigma_sq,
            device=expert_predictions.device,
            dtype=expert_predictions.dtype,
        ).clamp_min(eps)

        y_expanded = y.unsqueeze(1).expand_as(expert_predictions)  # (N, K)

        # Per-expert Gaussian NLL. full=True keeps the constant term.
        nll = F.gaussian_nll_loss(
            expert_predictions,
            y_expanded,
            sigma_sq.expand_as(expert_predictions),
            reduction="none",
            full=True,
        )  # (N, K)

        loglik = -nll

    else:
        y_expanded = y.unsqueeze(1).expand_as(expert_predictions)  # (N, K)

        # Treat expert outputs as logits for numerical stability.
        loglik = -F.binary_cross_entropy_with_logits(
            expert_predictions,
            y_expanded,
            reduction="none",
        )  # (N, K)

    logw = torch.log(w)

    return -torch.logsumexp(logw + loglik, dim=1).mean()


# ------------------------------------------------------------
# (B) Expected expert loss under gating
# ------------------------------------------------------------
def expected_expert_loss(batch, moe_model):
    """
    Expected per-expert loss under the gating distribution:

        E_{k ~ pi(x)} [ loss_k ]

    This is NOT a true likelihood. It is an upper bound / cooperative objective.
    """
    eps = moe_model.eps

    y = batch["phenotype"].float().view(-1)  # (N,)
    expert_predictions = moe_model._combined_linear_predictor(batch)  # (N, K)
    w = moe_model.gate_forward(batch)  # (N, K)

    if moe_model.family == "gaussian":
        sigma_sq = moe_model.sigma2 if moe_model.sigma2 is not None else 1.0
        sigma_sq = torch.as_tensor(
            sigma_sq,
            device=expert_predictions.device,
            dtype=expert_predictions.dtype,
        ).clamp_min(eps)

        y_expanded = y.unsqueeze(1).expand_as(expert_predictions)  # (N, K)

        losses = F.gaussian_nll_loss(
            expert_predictions,
            y_expanded,
            sigma_sq.expand_as(expert_predictions),
            reduction="none",
            full=True,
        )  # (N, K)

    else:
        y_expanded = y.unsqueeze(1).expand_as(expert_predictions)  # (N, K)

        losses = F.binary_cross_entropy_with_logits(
            expert_predictions,
            y_expanded,
            reduction="none",
        )  # (N, K)

    return (w * losses).sum(dim=1).mean()


# ------------------------------------------------------------
# (C) Ensemble prediction loss (mixture-of-means)
# ------------------------------------------------------------


def ensemble_prediction_loss(batch, moe_model):
    """
    Loss applied after averaging predictions:

        loss(y, sum_k pi_k(x) f_k(x))

    This is NOT a mixture model likelihood.
    """
    eps = moe_model.eps

    y = batch["phenotype"].float().view(-1)  # (N,)
    w = moe_model.gate_forward(batch)  # (N, K)
    expert_predictions = moe_model._combined_linear_predictor(batch)  # (N, K)

    if moe_model.family == "gaussian":
        pred = (w * expert_predictions).sum(dim=1)  # (N,)
        return F.mse_loss(pred, y)

    else:
        if moe_model.binomial_mixing_level == "logit":
            logits = (w * expert_predictions).sum(dim=1)  # (N,)
            return F.binary_cross_entropy_with_logits(logits, y)
        else:
            pred = (w * torch.sigmoid(expert_predictions)).sum(dim=1)  # (N,)
            pred = pred.clamp(eps, 1 - eps)
            return F.binary_cross_entropy(pred, y)


# ------------------------------------------------------------
# (D) Hybrid prediction loss
# Combines both ensemble loss + expected expert loss
# ------------------------------------------------------------


def ensemble_prediction_loss_with_specialization(
    batch, moe_model, specialization_weight=0.8
):
    assert 0.0 <= specialization_weight <= 1.0

    eps = moe_model.eps

    y = batch["phenotype"].float().view(-1)  # (N,)
    w = moe_model.gate_forward(batch)  # (N, K)
    expert_predictions = moe_model._combined_linear_predictor(batch)  # (N, K)

    # --------------------------------------------
    # Ensemble loss: loss(y, sum_k pi_k f_k)
    # --------------------------------------------
    if moe_model.family == "gaussian":
        pred = (w * expert_predictions).sum(dim=1)  # (N,)
        ensemble_loss = F.mse_loss(pred, y)

    else:
        if moe_model.binomial_mixing_level == "logit":
            logits = (w * expert_predictions).sum(dim=1)  # (N,)
            ensemble_loss = F.binary_cross_entropy_with_logits(logits, y)
        else:
            pred = (w * torch.sigmoid(expert_predictions)).sum(dim=1)  # (N,)
            pred = pred.clamp(eps, 1 - eps)
            ensemble_loss = F.binary_cross_entropy(pred, y)

    # --------------------------------------------
    # Specialization loss: expected expert loss
    # --------------------------------------------
    if moe_model.family == "gaussian":
        y_expanded = y.unsqueeze(1)  # (N, 1)
        expert_losses = F.mse_loss(
            expert_predictions,
            y_expanded.expand_as(expert_predictions),
            reduction="none",
        )  # (N, K)

        # Scale with sigma2 (if tuned):
        sigma_sq = moe_model.sigma2 if moe_model.sigma2 is not None else 1.0
        sigma_sq = torch.as_tensor(sigma_sq, device=expert_losses.device).clamp_min(eps)
        expert_losses = expert_losses / sigma_sq

    else:
        y_expanded = y.unsqueeze(1).expand_as(expert_predictions)  # (N, K)

        if moe_model.binomial_mixing_level == "logit":
            expert_losses = F.binary_cross_entropy_with_logits(
                expert_predictions, y_expanded, reduction="none"
            )  # (N, K)
        else:
            expert_probs = torch.sigmoid(expert_predictions)
            expert_probs = expert_probs.clamp(eps, 1 - eps)
            expert_losses = F.binary_cross_entropy(
                expert_probs, y_expanded, reduction="none"
            )  # (N, K)

    expected_expert_loss = (w * expert_losses).sum(dim=1).mean()

    # --------------------------------------------
    # Combine the two losses
    # --------------------------------------------
    return (
        1.0 - specialization_weight
    ) * ensemble_loss + specialization_weight * expected_expert_loss


# ------------------------------------------------------------
# (E) Penalties (mainly on the gating model output)
# ------------------------------------------------------------


def load_balance_penalty(expert_weights, eps=1e-8):

    expert_weights = expert_weights.clamp(min=eps, max=1.0 - eps)
    mean_usage = expert_weights.mean(dim=0)  # (K,)
    # Uniform distribution:
    target = torch.full_like(mean_usage, 1.0 / expert_weights.shape[1])

    lb_loss = ((mean_usage - target) ** 2).mean()

    return lb_loss


def entropy_penalty(expert_weights, eps=1e-8):

    p = expert_weights.clamp(min=eps, max=1.0 - eps)
    ent = -(p * p.log()).sum(dim=1).mean()
    return -ent
